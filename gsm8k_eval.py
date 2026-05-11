"""
CORA GSM8K Evaluation
─────────────────────
Benchmark: GSM8K (Grade School Math 8K) — openai/gsm8k
Evaluates CORA routing on math word problems.

Metrics computed:
  1. Accuracy          — exact final answer match
  2. Cost Reduction    — vs always using Tier 4
  3. Latency           — avg seconds per call
  4. RS Calibration Error (ECE) — routing score vs actual accuracy

No judge model needed: final numeric answer is extracted and compared exactly.
"""

import os
import re
import time
import asyncio
import random
import httpx
import pandas as pd
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

from cognitive import analyze_prompt
from complexity_score import score_to_tier

load_dotenv()

# ── Model config ──────────────────────────────────────────────────────────────

TIER_CONFIG = {
    "Tier 0": [
        {"model": "google/gemma-3n-e4b-it", "key": os.getenv("NVIDIA_GEMMA_API_KEY")},
    ],
    "Tier 1": [
        {"model": "mistralai/mistral-nemotron", "key": os.getenv("NVIDIA_NEMOTRON_API_KEY")},
    ],
    "Tier 2": [
        {"model": "mistralai/mistral-nemotron", "key": os.getenv("NVIDIA_NEMOTRON_API_KEY")},
    ],
    "Tier 3": [
        {"model": "nvidia/llama-3.3-nemotron-super-49b-v1", "key": os.getenv("NVIDIA_LLAMA_SUPER_API_KEY")},
    ],
    "Tier 4": [
        {"model": "nvidia/llama-3.3-nemotron-super-49b-v1", "key": os.getenv("NVIDIA_LLAMA_SUPER_API_KEY")},
    ],
}

TIER_COST = {
    "Tier 0": 1,
    "Tier 1": 4,
    "Tier 2": 12,
    "Tier 3": 30,
    "Tier 4": 80,
}

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

GSM8K_SYSTEM = (
    "You are a math tutor. Solve the problem step by step, "
    "then write your final numeric answer on the last line as: #### <number>"
)

def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the numeric answer after #### (GSM8K format)."""
    if text is None: return None
    # 1. Try GSM8K-style #### marker first
    m = re.search(r"####\s*[\$]?\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip().rstrip(".")
        
    # 2. Try common "Answer:" or "answer is" patterns
    m = re.search(r"(?:answer is|Answer:)\s*[\$]?\s*([\d,\.\-]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip().rstrip(".")

    # 3. Fallback: find the absolute last number mentioned in the text
    nums = re.findall(r"(\d+(?:,\d+)*(?:\.\d+)?)", text)
    if nums:
        return nums[-1].replace(",", "")
        
    return None


def normalize_answer(ans: str | None) -> str | None:
    """Normalise to float string for comparison."""
    if ans is None:
        return None
    try:
        return str(float(ans.replace(",", "")))
    except ValueError:
        return ans.strip().lower()


async def call_nim(client: httpx.AsyncClient, tier: str, prompt: str) -> tuple[str, float]:
    """Call NIM endpoint with failover support."""
    configs = TIER_CONFIG.get(tier, [])
    t0 = time.time()
    
    for config in configs:
        model = config["model"]
        api_key = config["key"]
        
        for attempt in range(2): # 2 attempts per model
            try:
                r = await client.post(
                    NIM_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": GSM8K_SYSTEM},
                            {"role": "user",   "content": prompt},
                        ],
                        "max_tokens": 512,
                        "temperature": 0.0,
                    },
                    timeout=15,
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip(), time.time() - t0
                
                # If degraded or not found, try next model in tier
                if r.status_code in [400, 404, 500, 503]:
                    print(f"  [FAILOVER] {tier} model '{model}' failed ({r.status_code}). Trying backup...")
                    break # Break inner attempt loop, go to next config
                    
                if r.status_code == 429 and attempt < 1:
                    await asyncio.sleep(5)
                    continue
            except Exception as e:
                print(f"  [FAILOVER] {tier} model '{model}' error: {e}. Trying backup...")
                break
                
    return "ERROR_all_models_failed", time.time() - t0


# ── Core evaluation ───────────────────────────────────────────────────────────

async def evaluate_gsm8k(num_samples: int = 50):
    print("Loading GSM8K dataset (test split)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  Dataset size: {len(dataset)} problems")

    # Sample
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    subset  = dataset.select(indices)

    print(f"\nEvaluating {len(subset)} GSM8K problems through CORA routing...\n")

    results = []

    async with httpx.AsyncClient() as client:
        for i, item in enumerate(subset):
            question   = item["question"].strip()
            # GSM8K ground truth answer always ends with #### <number>
            ref_answer = extract_gsm8k_answer(item["answer"])

            # 1. CORA routing
            profile            = analyze_prompt(question)
            tier, score, budget = score_to_tier(profile, question)

            # 2. LLM call
            response, latency = await call_nim(client, tier, question)

            # 3. Extract model's answer
            pred_answer = extract_gsm8k_answer(response)

            # 4. Exact-match comparison (normalised float)
            is_correct = (
                normalize_answer(pred_answer) == normalize_answer(ref_answer)
            )

            results.append({
                "question":   question[:80],
                "tier":       tier,
                "score":      score,
                "latency":    latency,
                "correct":    is_correct,
                "pred":       pred_answer,
                "target":     ref_answer,
                "cost":       TIER_COST[tier],
                "task_type":  profile.task_type.value,
            })

            status = "OK" if is_correct else "FAIL"
            print(f"  [{i+1:>3}/{len(subset)}] {tier} | {status} | pred={pred_answer} ref={ref_answer} | {question[:45]}...")

            # Small delay every 10 samples to respect rate limits
            if (i + 1) % 10 == 0:
                await asyncio.sleep(2)

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_and_print_metrics(results: list[dict]):
    df = pd.DataFrame(results)

    # 1. Accuracy
    accuracy = df["correct"].mean() * 100

    # 2. Cost Reduction vs always-Tier-4
    always_top_cost = TIER_COST["Tier 4"] * len(df)
    cora_cost       = df["cost"].sum()
    cost_reduction  = (1 - cora_cost / always_top_cost) * 100

    # 3. Average Latency
    avg_latency = df["latency"].mean()

    # 4. RS Calibration Error (ECE-style)
    #    Treat normalised CORA routing score as "confidence" in the routing decision.
    #    Measure |avg_score_in_bin - accuracy_in_bin| weighted by bin size.
    df["norm_score"] = df["score"] / 100.0
    bins = np.linspace(0, 1, 6)
    df["bin"] = pd.cut(df["norm_score"], bins=bins, include_lowest=True)

    ece = 0.0
    n   = len(df)
    bin_table = []
    for b in sorted(df["bin"].unique(), key=lambda x: x.left if not pd.isna(x) else -1):
        if pd.isna(b):
            continue
        bin_df = df[df["bin"] == b]
        if len(bin_df) == 0:
            continue
        bin_acc  = bin_df["correct"].mean()
        bin_conf = bin_df["norm_score"].mean()
        # In CORA, norm_score represents difficulty (0=easy, 1=hard).
        # To calculate calibration, we compare accuracy with confidence (1 - difficulty).
        confidence = 1.0 - bin_conf
        weight   = len(bin_df) / n
        gap      = abs(bin_acc - confidence)
        ece     += weight * gap
        bin_table.append({
            "bin":      str(b),
            "n":        len(bin_df),
            "accuracy": bin_acc,
            "conf":     confidence,
            "gap":      gap,
        })

    rs_cal_error = ece * 100

    # ── Tier distribution ──────────────────────────────────────────────
    tier_dist = df["tier"].value_counts().sort_index()

    print("\n" + "=" * 55)
    print("  CORA × GSM8K — EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Samples evaluated   : {len(df)}")
    print(f"  Accuracy            : {accuracy:.2f}%")
    print(f"  Cost Reduction      : {cost_reduction:.2f}%  (vs always Tier 4)")
    print(f"  Avg Latency         : {avg_latency:.2f} s/call")
    print(f"  RS Calibration Error: {rs_cal_error:.2f}%")
    print("=" * 55)

    print("\n  Tier Distribution:")
    for tier, count in tier_dist.items():
        pct = count / len(df) * 100
        bar = "#" * int(pct / 4)
        print(f"    {tier}: {bar} {pct:.1f}% ({count})")

    print("\n  RS Calibration Bins:")
    print(f"  {'Bin':<18} {'n':>4} {'Accuracy':>9} {'RoutScore':>10} {'|Gap|':>7}")
    print("  " + "-" * 52)
    for row in bin_table:
        print(f"  {row['bin']:<18} {row['n']:>4} {row['accuracy']*100:>8.1f}%"
              f" {row['conf']*100:>9.1f}% {row['gap']*100:>6.1f}%")

    print("\n  Task-type accuracy:")
    for task, grp in df.groupby("task_type"):
        acc = grp["correct"].mean() * 100
        print(f"    {task:<22}: {acc:.1f}%  (n={len(grp)})")

    print("=" * 55)

    # Save per-sample results
    df.to_csv("gsm8k_results.csv", index=False)
    print("\n  Full results saved to gsm8k_results.csv")

    # Append summary row to shared benchmark_summary.csv
    import csv as _csv, datetime as _dt
    summary_path = "benchmark_summary.csv"
    summary_row = {
        "benchmark":       "GSM8K",
        "samples":         len(df),
        "accuracy_pct":    round(accuracy, 2),
        "cost_reduction_pct": round(cost_reduction, 2),
        "avg_latency_s":   round(avg_latency, 3),
        "rs_cal_error_pct": round(rs_cal_error, 2),
        "timestamp":       _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_header = not __import__("os").path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=summary_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(f"  Summary appended to {summary_path}")

    return {
        "accuracy":        accuracy,
        "cost_reduction":  cost_reduction,
        "avg_latency":     avg_latency,
        "rs_cal_error":    rs_cal_error,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    random.seed(42)
    results = await evaluate_gsm8k(num_samples=50)
    compute_and_print_metrics(results)


if __name__ == "__main__":
    asyncio.run(main())
