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

async def call_nim_with_latency(client: httpx.AsyncClient, tier: str, prompt: str, max_tokens: int = 10):
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
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    },
                    timeout=15,
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip(), time.time() - t0
                
                # If degraded or not found, try next model in tier
                if r.status_code in [400, 404, 500, 503]:
                    break
                    
                if r.status_code == 429 and attempt < 1:
                    await asyncio.sleep(5)
                    continue
            except Exception:
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
    for char in response_text:
        if char in ["A", "B", "C", "D"]:
            return char
    return None

async def evaluate_mmlu():
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df_all = dataset.to_pandas()
    
    print("Sampling 100 questions from each of the 57 subjects...")
    # Group by subject and sample 100 from each
    subset_df = df_all.groupby("subject").apply(lambda x: x.sample(n=min(100, len(x)), random_state=42)).reset_index(drop=True)
    num_samples = len(subset_df)
    
    # Convert back to list of dicts for iteration
    subset = subset_df.to_dict("records")
    
    results = []
    
    print(f"Evaluating {num_samples} MMLU prompts across 57 fields...")
    async with httpx.AsyncClient() as client:
        for i, item in enumerate(subset):
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
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{num_samples}")
                
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

    df.to_csv("mmlu_results.csv", index=False)
    print("\nResults saved to mmlu_results.csv")

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
    asyncio.run(evaluate_mmlu())
