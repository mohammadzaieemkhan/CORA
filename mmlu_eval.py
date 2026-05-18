import os
import time
import json
import asyncio
import random
import httpx
import pandas as pd
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

# Import CORA modules
from cognitive import analyze_prompt
from complexity_score import score_to_tier

load_dotenv()

TIER_CONFIG = {
    "Tier 0": [
        {"model": "nvidia/nemotron-mini-4b-instruct", "key": os.getenv("NVIDIA_NEMOTRON_MINI_API_KEY")},
        {"model": "google/gemma-3n-e4b-it", "key": os.getenv("NVIDIA_GEMMA_3N_API_KEY")},
    ],
    "Tier 1": [
        {"model": "nvidia/nvidia-nemotron-nano-9b-v2", "key": os.getenv("NVIDIA_NEMOTRON_NANO_9B_API_KEY")},
    ],
    "Tier 2": [
        {"model": "nvidia/nemotron-3-nano-30b-a3b", "key": os.getenv("NVIDIA_NEMOTRON_NANO_30B_API_KEY")},
    ],
    "Tier 3": [
        {"model": "nvidia/nemotron-3-super-120b-a12b", "key": os.getenv("NVIDIA_NEMOTRON_SUPER_API_KEY")},
        {"model": "mistralai/mistral-medium-3.5-128b", "key": os.getenv("NVIDIA_MISTRAL_MEDIUM_API_KEY")},
    ],
    "Tier 4": [
        {"model": "qwen/qwen3-coder-480b-a35b-instruct", "key": os.getenv("NVIDIA_QWEN3_CODER_API_KEY")},
        {"model": "qwen/qwen3.5-397b-a17b", "key": os.getenv("NVIDIA_QWEN3_5_API_KEY")},
        {"model": "nvidia/nemotron-3-super-120b-a12b", "key": "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"},
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

async def call_nim_with_latency(client: httpx.AsyncClient, tier: str, prompt: str, max_tokens: int = 512):
    """Call NIM endpoint with failover support."""
    configs = TIER_CONFIG.get(tier, [])
    t0 = time.time()
    
    for config in configs:
        model = config["model"]
        api_key = config["key"]
        if not api_key:
            print(f"  [FAILOVER] No API key configured for {tier} model '{model}'. Skipping...")
            continue
            
        # Optimize thinking budgets based on the model ID to allow natural, capped reasoning
        extra_body = {}
        if "nano-9b" in model:
            extra_body = {
                "max_thinking_tokens": 512,
            }
        elif "nano-30b" in model:
            extra_body = {
                "reasoning_budget": 512,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            
        for attempt in range(2): # 2 attempts per model
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a multiple-choice question solver. You must answer exactly with a single letter (A, B, C, or D) representing the correct choice. Do not explain, repeat, or preface."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                }
                if extra_body:
                    payload.update(extra_body)
                    
                r = await client.post(
                    NIM_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=60, # Increased to 60 to prevent timeouts
                )
                if r.status_code == 200:
                    res_json = r.json()
                    message = res_json["choices"][0]["message"]
                    content = message.get("content")
                    if content is None:
                        content = message.get("reasoning_content") or ""
                    return content.strip(), time.time() - t0
                
                # If degraded or not found, try next model in tier
                if r.status_code in [400, 404, 500, 503]:
                    print(f"  [FAILOVER] {tier} model '{model}' failed ({r.status_code}: {r.text[:200]}). Trying backup...")
                    break
                    
                if r.status_code == 429 and attempt < 1:
                    await asyncio.sleep(5)
                    continue
            except Exception as e:
                print(f"  [FAILOVER] {tier} model '{model}' error: {e}. Trying backup...")
                break
                
    return "ERROR_all_models_failed", time.time() - t0

def format_mmlu_prompt(question, choices):
    prompt = f"{question}\n"
    labels = ["A", "B", "C", "D"]
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer exactly with a single letter (A, B, C, or D):"
    return prompt

def extract_answer(response_text):
    if not response_text:
        return None
    text = response_text.strip()
    
    # 1. Check if the response is exactly one letter A, B, C, D
    if len(text) == 1 and text.upper() in ["A", "B", "C", "D"]:
        return text.upper()
        
    # 2. Search for explicit patterns like "choice is A", "option B", "correct: C", "answer is D"
    import re
    matches = re.findall(r"(?:answer|option|choice|letter|is|correct|be)\s*[:\-\s]*([A-D])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
        
    # 3. Search for a bracketed/dotted answer at the end like "(A)" or "A." or simply ending with "A"
    matches_end = re.findall(r"\b([A-D])[\.\s]*$", text)
    if matches_end:
        return matches_end[-1].upper()
        
    # 4. Fallback: Scan from right-to-left for the first capital letter A, B, C, or D (representing their final output choice)
    for char in reversed(text):
        if char in ["A", "B", "C", "D"]:
            return char
    return None

async def evaluate_mmlu(num_samples: int = 50):
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df_all = dataset.to_pandas()
    
    # Standardize sampling based on num_samples to keep evaluation fast & flexible
    if num_samples < 100:
        subset_df = df_all.sample(n=min(num_samples, len(df_all)), random_state=42).reset_index(drop=True)
    else:
        samples_per_subject = max(1, num_samples // 57)
        print(f"Sampling {samples_per_subject} questions from each of the 57 subjects...")
        subset_df = df_all.groupby("subject").apply(lambda x: x.sample(n=min(samples_per_subject, len(x)), random_state=42)).reset_index(drop=True)
        
    num_samples = len(subset_df)
    
    # Convert back to list of dicts for iteration
    subset = subset_df.to_dict("records")
    
    results = []
    sem = asyncio.Semaphore(15)  # 15 concurrent requests to safely avoid 429/503 API rate limits

    async def evaluate_item(client, item):
        async with sem:
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            correct_label = ["A", "B", "C", "D"][answer_idx]
            
            prompt = format_mmlu_prompt(question, choices)
            
            # 1. Routing
            profile = analyze_prompt(prompt)
            tier, score, budget = score_to_tier(profile, prompt)
            
            # 2. Call Model
            response_text, latency = await call_nim_with_latency(client, tier, prompt)
            
            # 3. Accuracy evaluation
            pred_label = extract_answer(response_text)
            is_correct = (pred_label == correct_label)
            
            results.append({
                "question": question[:50],
                "tier": tier,
                "score": score,
                "latency": latency,
                "correct": is_correct,
                "pred": pred_label,
                "target": correct_label,
                "cost": TIER_COST[tier]
            })
            
            if len(results) % 50 == 0:
                print(f"Processed {len(results)}/{num_samples}")

    print(f"Evaluating {num_samples} MMLU prompts across fields...")
    async with httpx.AsyncClient() as client:
        tasks = [evaluate_item(client, item) for item in subset]
        await asyncio.gather(*tasks)
                
    df = pd.DataFrame(results)
    
    # Calculate Metrics
    accuracy = df["correct"].mean() * 100
    avg_latency = df["latency"].mean()
    
    always_tier4_cost = TIER_COST["Tier 4"] * len(df)
    cora_cost = df["cost"].sum()
    cost_reduction = (1 - cora_cost / always_tier4_cost) * 100
    
    # RS Calibration Error (Routing Score Calibration Error)
    df["norm_score"] = df["score"] / 100.0
    bins = np.linspace(0, 1, 6)
    df["bin"] = pd.cut(df["norm_score"], bins=bins, include_lowest=True)
    
    ece = 0
    total_samples = len(df)
    for b in df["bin"].unique():
        if pd.isna(b):
            continue
        bin_data = df[df["bin"] == b]
        if len(bin_data) > 0:
            bin_acc = bin_data["correct"].mean()
            bin_conf = bin_data["norm_score"].apply(lambda x: 1 - x).mean()
            weight = len(bin_data) / total_samples
            ece += weight * abs(bin_acc - bin_conf)
            
    rs_cal_error = ece * 100
    
    print("\n" + "="*50)
    print(" MMLU Evaluation Results")
    print("="*50)
    print(f"  Accuracy           : {accuracy:.2f}%")
    print(f"  Cost Reduction     : {cost_reduction:.2f}% (vs Tier 4)")
    print(f"  Average Latency    : {avg_latency:.2f} s")
    print(f"  RS Calibration Error: {rs_cal_error:.2f}%")
    print("="*50)

    df.to_csv("mmluresult.csv", index=False)
    print("\nResults saved to mmluresult.csv")

    # Append summary row to shared benchmark_summary.csv
    import csv as _csv, datetime as _dt
    summary_path = "benchmark_summary.csv"
    summary_row = {
        "benchmark":          "MMLU",
        "samples":            len(df),
        "accuracy_pct":       round(accuracy, 2),
        "cost_reduction_pct": round(cost_reduction, 2),
        "avg_latency_s":      round(avg_latency, 3),
        "rs_cal_error_pct":   round(rs_cal_error, 2),
        "timestamp":          _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_header = not __import__("os").path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=summary_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(f"Summary appended to {summary_path}")

if __name__ == "__main__":
    import sys
    samples = 50
    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            pass
    asyncio.run(evaluate_mmlu(samples))
