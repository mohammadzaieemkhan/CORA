"""
CORA Comprehensive Model Verification & Benchmark Suite
Checks all LLM models, measures precise response latency, tests end-to-end tier routing,
and displays models ranked in ascending order by response time for the demo.
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(Path(__file__).parent / ".env")

from llm_providers import (
    gemma_4_26b,
    gemma_4_26b_openrouter,
    gemini_3_5_flash_lite,
    gemma_2_9b_openrouter,
    muse_glimmer_30b,
    nemotron_3_5_lightning_30b,
    nemotron_super_120b,
    glm_5_3_flash,
    nemotron_3_ultra_550b,
    deepseek_v4_1_flash,
    kimi_k3,
    call_llm,
)
from llm_providers.prompt_optimizer import optimize_prompt

TEST_PROMPT = "What is 2 + 2? Answer in one word."

ACTIVE_TIER_MODELS = [
    ("Tier 0 Primary",  gemma_4_26b),
    ("Tier 1 Primary",  gemini_3_5_flash_lite),
    ("Tier 2 Primary",  nemotron_super_120b),
    ("Tier 3 Primary",  nemotron_super_120b),
    ("Tier 4 Primary",  nemotron_3_ultra_550b),
    ("Tier 4 Fallback", kimi_k3),
    ("Tier 2 Fallback", muse_glimmer_30b),
]

SECONDARY_DIAGNOSTIC_MODELS = [
    ("Tier 0 Backup",   gemma_4_26b_openrouter),
    ("Tier 1 Backup",   gemma_2_9b_openrouter),
    ("Tier 2 Backup",   nemotron_3_5_lightning_30b),
    ("Tier 3 Backup",   deepseek_v4_1_flash),
    ("Tier 3 Backup 2", glm_5_3_flash),
]


async def check_all_models():
    print("=" * 110, flush=True)
    print("  PHASE 1: BENCHMARKING ACTIVE CORA MODELS (MEASURING LATENCY)", flush=True)
    print("=" * 110, flush=True)

    results = []

    for role, mod in ACTIVE_TIER_MODELS:
        t0 = time.time()
        name = getattr(mod, "DISPLAY_NAME", mod.__name__)
        key_env = getattr(mod, "API_KEY_ENV", "CUSTOM")
        key = getattr(mod, "get_api_key", lambda: "")()

        if not key:
            print(f"  [MISSING KEY] {role:17s} | {name:32s} | Env: {key_env}", flush=True)
            results.append((name, role, "MISSING_KEY", 0.0, "Key not set in .env"))
            continue

        try:
            ans = await asyncio.wait_for(mod.call(TEST_PROMPT), timeout=25.0)
            elapsed = round(time.time() - t0, 3)
            clean_ans = str(ans or "").replace("\n", " ").strip()[:45]
            print(f"  [PASS]        {role:17s} | {name:32s} | {elapsed:6.3f}s | Output: {clean_ans}", flush=True)
            results.append((name, role, "PASS", elapsed, clean_ans))
        except asyncio.TimeoutError:
            elapsed = round(time.time() - t0, 3)
            print(f"  [TIMEOUT]     {role:17s} | {name:32s} | {elapsed:6.3f}s | Exceeded 25s limit", flush=True)
            results.append((name, role, "TIMEOUT", elapsed, "Exceeded 25s limit"))
        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            err_msg = str(e)[:70]
            print(f"  [FAIL]        {role:17s} | {name:32s} | {elapsed:6.3f}s | Error: {err_msg}", flush=True)
            results.append((name, role, "FAIL", elapsed, err_msg))

    # Test Prompt Optimizer Engine
    t0 = time.time()
    try:
        opt_ans = await asyncio.wait_for(
            optimize_prompt("Could you please tell me what the capital of Spain is?"),
            timeout=15.0
        )
        elapsed = round(time.time() - t0, 3)
        clean_ans = str(opt_ans or "").replace("\n", " ").strip()[:45]
        print(f"  [PASS]        Optimizer Engine  | Google Gemini Flash Lite (Opt)   | {elapsed:6.3f}s | Output: {clean_ans}", flush=True)
        results.append(("Google Gemini Flash Lite (Opt)", "Optimizer Engine", "PASS", elapsed, clean_ans))
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        err_msg = str(e)[:70]
        print(f"  [FAIL]        Optimizer Engine  | Google Gemini Flash Lite (Opt)   | {elapsed:6.3f}s | Error: {err_msg}", flush=True)
        results.append(("Google Gemini Flash Lite (Opt)", "Optimizer Engine", "FAIL", elapsed, err_msg))

    return results


async def check_tier_routing():
    print("\n" + "=" * 110, flush=True)
    print("  PHASE 2: TESTING END-TO-END TIER ROUTING (WITH REAL-TIME FALLBACK)", flush=True)
    print("=" * 110, flush=True)

    tier_prompts = [
        ("Tier 0", "Say hello in one word."),
        ("Tier 1", "What is the boiling point of water in Celsius? Number only."),
        ("Tier 2", "Compare merge sort and quick sort time complexity in one sentence."),
        ("Tier 3", "Write a Python function to reverse a string in one line: def rev(s): ..."),
        ("Tier 4", "Solve step-by-step: If 3x + 5 = 20, what is x? Provide final number only."),
    ]

    routing_results = []
    for tier, prompt in tier_prompts:
        t0 = time.time()
        try:
            ans, resolved_model = await asyncio.wait_for(
                call_llm(tier, prompt),
                timeout=30.0
            )
            elapsed = round(time.time() - t0, 3)
            clean_ans = str(ans or "").replace("\n", " ").strip()[:45]
            print(f"  [ROUTED OK]   {tier:8s} -> Served by: {resolved_model:32s} | {elapsed:6.3f}s | Output: {clean_ans}", flush=True)
            routing_results.append((tier, resolved_model, elapsed, "PASS"))
        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            print(f"  [ROUTING ERR] {tier:8s} -> Failed in {elapsed:6.3f}s | {e}", flush=True)
            routing_results.append((tier, "None", elapsed, "FAIL"))

    print("=" * 110, flush=True)
    return routing_results


def print_ascending_benchmark_table(results):
    print("\n" + "=" * 110, flush=True)
    print("  DEMO SHOWCASE: ALL ACTIVE CORA MODELS RANKED BY SPEED (ASCENDING ORDER)", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'Rank':<6} | {'Response Time':<15} | {'Status':<8} | {'Tier / Role':<20} | {'Model Name'}", flush=True)
    print("  " + "-" * 104, flush=True)

    # Sort primarily by PASS vs FAIL, then by elapsed time ascending
    sorted_results = sorted(results, key=lambda x: (x[2] != "PASS", x[3]))

    for idx, (name, role, status, elapsed, output) in enumerate(sorted_results, 1):
        status_tag = f"[{status}]"
        time_str = f"{elapsed:6.3f}s" if status == "PASS" else "N/A"
        print(f"  #{idx:<5} | {time_str:<15} | {status_tag:<8} | {role:<20} | {name}", flush=True)

    print("=" * 110 + "\n", flush=True)


async def main():
    results = await check_all_models()
    print_ascending_benchmark_table(results)
    await check_tier_routing()


if __name__ == "__main__":
    asyncio.run(main())
