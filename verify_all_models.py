"""Comprehensive diagnostic to verify every model and tier in CORA."""
import asyncio
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

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

MODULES_TO_TEST = [
    ("Tier 0 Primary",  gemma_4_26b),
    ("Tier 0 Fallback", gemma_4_26b_openrouter),
    ("Tier 1 Primary",  gemini_3_5_flash_lite),
    ("Tier 1 Fallback", gemma_2_9b_openrouter),
    ("Tier 2 Primary",  muse_glimmer_30b),
    ("Tier 2 Fallback", nemotron_3_5_lightning_30b),
    ("Tier 3 Primary",  nemotron_super_120b),
    ("Tier 3 Fallback", deepseek_v4_1_flash),
    ("Tier 4 Primary",  nemotron_3_ultra_550b),
    ("Tier 4 Fallback", kimi_k3),
]

async def check_individual_models():
    print("=" * 105, flush=True)
    print("  PHASE 1: TESTING EVERY CONFIGURED MODEL INDIVIDUALLY", flush=True)
    print("=" * 105, flush=True)

    results = []

    for role, mod in MODULES_TO_TEST:
        t0 = time.time()
        name = getattr(mod, "DISPLAY_NAME", mod.__name__)
        key_env = getattr(mod, "API_KEY_ENV", "CUSTOM")
        key = getattr(mod, "get_api_key", lambda: "")()

        if not key:
            print(f"  [MISSING KEY] {role:17s} | {name:28s} | Env: {key_env}", flush=True)
            results.append((role, name, "MISSING_KEY", 0.0, "Key not set in .env"))
            continue

        try:
            # Run with a 25s timeout per model
            ans = await asyncio.wait_for(mod.call(TEST_PROMPT), timeout=25.0)
            elapsed = round(time.time() - t0, 2)
            clean_ans = str(ans or "").replace("\n", " ").strip()[:50]
            print(f"  [PASS]        {role:17s} | {name:28s} | {elapsed:5.2f}s | Output: {clean_ans}", flush=True)
            results.append((role, name, "PASS", elapsed, clean_ans))
        except asyncio.TimeoutError:
            elapsed = round(time.time() - t0, 2)
            print(f"  [TIMEOUT]     {role:17s} | {name:28s} | {elapsed:5.2f}s | Exceeded 25s limit", flush=True)
            results.append((role, name, "TIMEOUT", elapsed, "Exceeded 25s limit"))
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            err_msg = str(e)[:75]
            print(f"  [FAIL]        {role:17s} | {name:28s} | {elapsed:5.2f}s | Error: {err_msg}", flush=True)
            results.append((role, name, "FAIL", elapsed, err_msg))

    # Test Prompt Optimizer (Google Gemini)
    t0 = time.time()
    try:
        opt_ans = await asyncio.wait_for(
            optimize_prompt("Could you please tell me what the capital of Spain is?"),
            timeout=20.0
        )
        elapsed = round(time.time() - t0, 2)
        clean_ans = str(opt_ans or "").replace("\n", " ").strip()[:50]
        print(f"  [PASS]        Optimizer Engine  | Google Gemini Top-Tier       | {elapsed:5.2f}s | Output: {clean_ans}", flush=True)
        results.append(("Optimizer", "Google Gemini Top-Tier", "PASS", elapsed, clean_ans))
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        err_msg = str(e)[:75]
        print(f"  [FAIL]        Optimizer Engine  | Google Gemini Top-Tier       | {elapsed:5.2f}s | Error: {err_msg}", flush=True)
        results.append(("Optimizer", "Google Gemini Top-Tier", "FAIL", elapsed, err_msg))

    return results


async def check_tier_routing():
    print("\n" + "=" * 105, flush=True)
    print("  PHASE 2: TESTING END-TO-END TIER ROUTING (WITH AUTOMATIC FALLBACK)", flush=True)
    print("=" * 105, flush=True)

    tiers = ["Tier 0", "Tier 1", "Tier 2", "Tier 3", "Tier 4"]
    for tier in tiers:
        t0 = time.time()
        try:
            ans, resolved_model = await asyncio.wait_for(
                call_llm(tier, "What is 5 * 5? Answer with only the number."),
                timeout=30.0
            )
            elapsed = round(time.time() - t0, 2)
            clean_ans = str(ans or "").replace("\n", " ").strip()[:40]
            print(f"  [ROUTED OK]   {tier:8s} -> Served by: {resolved_model:28s} | {elapsed:5.2f}s | Output: {clean_ans}", flush=True)
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            print(f"  [ROUTING ERR] {tier:8s} -> Failed in {elapsed:5.2f}s | {e}", flush=True)

    print("=" * 105 + "\n", flush=True)


async def main():
    await check_individual_models()
    await check_tier_routing()


if __name__ == "__main__":
    asyncio.run(main())
