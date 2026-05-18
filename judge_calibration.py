"""
CORA Calibration Judge
─────────────────────
Sends each prompt to its assigned tier AND one tier lower,
scores both responses using Nemotron Nano 30B as judge (0/1/2),
computes under-routing rate and accuracy metrics.
Judge model: nvidia/nemotron-3-nano-30b-a3b (via NVIDIA NIM)
"""

import json
import csv
import asyncio
import os
import random
from collections import defaultdict
from dotenv import load_dotenv
import httpx
from cognitive import analyze_prompt
from complexity_score import score_to_tier

load_dotenv()

# Per-model API keys matching .env configuration
TIER_KEYS = {
    "Tier 0": os.getenv("NVIDIA_NEMOTRON_MINI_API_KEY"),
    "Tier 1": os.getenv("NVIDIA_NEMOTRON_NANO_9B_API_KEY"),
    "Tier 2": os.getenv("NVIDIA_NEMOTRON_NANO_30B_API_KEY"),
    "Tier 3": os.getenv("NVIDIA_NEMOTRON_SUPER_API_KEY"),
    "Tier 4": os.getenv("NVIDIA_QWEN3_CODER_API_KEY"),
}
JUDGE_KEY = os.getenv("NVIDIA_NEMOTRON_NANO_30B_API_KEY")

TIER_MODELS = {
    "Tier 0": "nvidia/nemotron-mini-4b-instruct",
    "Tier 1": "nvidia/nvidia-nemotron-nano-9b-v2",
    "Tier 2": "nvidia/nemotron-3-nano-30b-a3b",
    "Tier 3": "nvidia/nemotron-3-super-120b-a12b",
    "Tier 4": "qwen/qwen3-coder-480b-a35b-instruct",
}

JUDGE_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
TIER_ORDER  = ["Tier 0", "Tier 1", "Tier 2", "Tier 3", "Tier 4"]
NIM_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"

JUDGE_PROMPT = """You are a strict quality evaluator for AI responses.

User prompt:
{prompt}

AI response:
{response}

Evaluate whether the AI response adequately addresses the user's request.
Consider: completeness, accuracy, relevance, and usefulness.

Score:
2 = Good. The response fully addresses the prompt. A typical user would be satisfied.
1 = Partial. The response is on-topic but has clear gaps, errors, or is too vague.
0 = Failure. The response is wrong, off-topic, refuses to answer, or is unusable.

IMPORTANT: Most competent responses deserve a 2. Reserve 1 for noticeably flawed responses and 0 for outright failures.

Reply with ONLY a single digit: 0, 1, or 2."""


# ── API helpers ───────────────────────────────────────────────────────────────

def _key_for_model(model: str) -> str:
    """Return the correct API key for a given model ID."""
    for tier, m in TIER_MODELS.items():
        if m == model:
            return TIER_KEYS.get(tier, JUDGE_KEY) or ""
    return JUDGE_KEY or ""


async def call_nim(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int = 500,
    retries: int = 2,
) -> str:
    """Call any NVIDIA NIM model with retry logic. Returns response text or ERROR_xxx."""
    api_key = _key_for_model(model)
    for attempt in range(retries + 1):
        try:
            r = await client.post(
                NIM_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429 and attempt < retries:
                await asyncio.sleep(5)  # rate limit — wait and retry
                continue
            return f"ERROR_{r.status_code}"
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            return f"ERROR_{e}"
    return "ERROR_max_retries"


async def judge_response(
    client: httpx.AsyncClient,
    prompt: str,
    response: str,
) -> int:
    """
    Use Mistral Large (free on NIM) to score a response 0/1/2.
    Returns 0 immediately if response is an error string.
    """
    if response.startswith("ERROR"):
        return 0

    raw = await call_nim(
        client,
        JUDGE_MODEL,
        JUDGE_PROMPT.format(prompt=prompt[:800], response=response[:1500]),
        max_tokens=1,
    )

    # parse — only accept 0, 1, 2
    raw = raw.strip()
    if raw in ("0", "1", "2"):
        return int(raw)

    # fallback: scan first char
    for ch in raw:
        if ch in ("0", "1", "2"):
            return int(ch)

    return 1  # default to marginal if judge is unclear


# ── Tier helpers ──────────────────────────────────────────────────────────────

def get_lower_tier(tier: str) -> str | None:
    idx = TIER_ORDER.index(tier)
    return TIER_ORDER[idx - 1] if idx > 0 else None


# ── Core prompt processor ─────────────────────────────────────────────────────

async def process_prompt(
    client: httpx.AsyncClient,
    prompt: str,
    idx: int,
) -> dict:
    """
    For one prompt:
      1. Score with CORA → assigned tier
      2. Call assigned tier model → response A
      3. Call one-tier-lower model → response B
      4. Judge both with Mistral Large
      5. Determine ground truth T* and whether CORA under-routed
    """
    profile = analyze_prompt(prompt)
    assigned_tier, cora_score, budget = score_to_tier(profile, prompt)
    lower_tier = get_lower_tier(assigned_tier)

    print(f"  [{idx+1:>3}] {assigned_tier} | task={profile.task_type.value:<15} | {prompt[:50]}")

    # ── Get responses ──────────────────────────────────────────────
    assigned_response = await call_nim(client, TIER_MODELS[assigned_tier], prompt)

    if lower_tier:
        lower_response = await call_nim(client, TIER_MODELS[lower_tier], prompt)
    else:
        lower_response = assigned_response  # Tier 0 has no lower

    # ── Judge both ────────────────────────────────────────────────
    assigned_judge = await judge_response(client, prompt, assigned_response)
    lower_judge    = await judge_response(client, prompt, lower_response)

    # ── Determine ground truth T* ─────────────────────────────────
    # Logic:
    #   - If assigned tier scores 2: it's correct (even if lower also scores 2,
    #     the assigned tier DID produce a good response).
    #   - If assigned tier scores < 2 but lower scores 2: over-routed
    #     (lower was cheaper and just as good).
    #   - If both score < 2: under-routed (need a higher tier).
    if assigned_judge == 2:
        if lower_tier and lower_judge == 2:
            # Both work — ground truth is the lower (cheaper) tier
            ground_truth = lower_tier
            under_routed = False
            correct      = False   # over-routed but not harmful
        else:
            # Assigned tier is the right one
            ground_truth = assigned_tier
            under_routed = False
            correct      = True
    else:
        # Assigned tier failed
        if lower_tier and lower_judge == 2:
            # Lower tier was actually fine — oddly over-routed AND failed
            ground_truth = lower_tier
            under_routed = False
            correct      = False
        else:
            # Both failed — need one tier higher than assigned
            idx = TIER_ORDER.index(assigned_tier)
            ground_truth = TIER_ORDER[min(idx + 1, 4)]
            under_routed = True
            correct      = False

    return {
        "prompt":                 prompt[:120],
        "task_type":              profile.task_type.value,
        "cora_score":             round(cora_score, 3),
        "budget_score":           budget,
        "assigned_tier":          assigned_tier,
        "lower_tier":             lower_tier or "N/A",
        "assigned_judge_score":   assigned_judge,
        "lower_judge_score":      lower_judge,
        "ground_truth_tier":      ground_truth,
        "correct_routing":        correct,
        "under_routed":           under_routed,
        "reasoning_depth":        profile.reasoning_depth,
        "domain_specificity":     profile.domain_specificity,
        "code_complexity":        profile.code_complexity,
        "creative_demand":        profile.creative_demand,
        "precision_required":     profile.precision_required,
        "structural_complexity":  profile.structural_complexity,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

async def run_batches(prompts: list[str], batch_size: int = 3) -> list[dict]:
    """
    Process prompts in small batches with a delay between batches
    to respect NIM rate limits (~40 req/min).
    """
    results = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            print(f"\nBatch {i//batch_size + 1} / {len(prompts)//batch_size + 1}")
            batch_results = await asyncio.gather(*[
                process_prompt(client, p, i + j)
                for j, p in enumerate(batch)
            ])
            results.extend(batch_results)
            if i + batch_size < len(prompts):
                await asyncio.sleep(3)  # 3s pause between batches
    return results


# ── Metrics printer ───────────────────────────────────────────────────────────

def print_metrics(results: list[dict]) -> None:
    total       = len(results)
    correct     = sum(1 for r in results if r["correct_routing"])
    under       = sum(1 for r in results if r["under_routed"])
    over        = total - correct - under

    within_one = 0
    for r in results:
        a = TIER_ORDER.index(r["assigned_tier"])
        g = TIER_ORDER.index(r["ground_truth_tier"])
        if abs(a - g) <= 1:
            within_one += 1

    print("\n" + "=" * 52)
    print("  CORA CALIBRATION RESULTS")
    print("=" * 52)
    print(f"  Prompts evaluated      : {total}")
    print(f"  Exact-tier accuracy    : {correct/total*100:>5.1f}%")
    print(f"  Within-1 accuracy      : {within_one/total*100:>5.1f}%")
    print(f"  Under-routing rate     : {under/total*100:>5.1f}%   <-- must be < 5%")
    print(f"  Over-routing rate      : {over/total*100:>5.1f}%   (costs more, not harmful)")
    print("=" * 52)

    # per task type
    task_stats = defaultdict(lambda: {"total": 0, "correct": 0, "under": 0})
    for r in results:
        t = r["task_type"]
        task_stats[t]["total"]   += 1
        if r["correct_routing"]: task_stats[t]["correct"] += 1
        if r["under_routed"]:    task_stats[t]["under"]   += 1

    print(f"\n  {'Task':<20} {'n':>4} {'Correct':>8} {'Under':>7}")
    print("  " + "-" * 42)
    for task, s in sorted(task_stats.items()):
        n = s["total"]
        print(f"  {task:<20} {n:>4} "
              f"{s['correct']/n*100:>7.0f}% "
              f"{s['under']/n*100:>6.0f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    # Load Alpaca dataset
    print("Loading calibration prompts...")
    with open("alpaca_data.json", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)

    # filter: not too short, not too long
    prompts = [
        item["instruction"].strip().replace("\n", " ")
        for item in data
        if 5 < len(item["instruction"].split()) < 150
    ][:100]

    print(f"Running judge on {len(prompts)} prompts using Nemotron Nano 30B as judge...")
    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Batch size  : 3 prompts  |  pause: 3s between batches\n")

    results = await run_batches(prompts, batch_size=3)

    # Save CSV
    csv_path = "judge_result(2).csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print_metrics(results)
    print(f"\n  Full results saved to {csv_path}")
    print("  Use judge_results.csv for ordinal regression in the next step.\n")


if __name__ == "__main__":
    asyncio.run(main())
