"""Quick diagnostic: test each NVIDIA API model endpoint."""
import asyncio, os, sys, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import httpx

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TEST_PROMPT = "Say hello in one word."

MODELS = [
    ("Tier 0  - Nemotron Mini 4B",    "nvidia/nemotron-mini-4b-instruct",          "NVIDIA_NEMOTRON_MINI_API_KEY"),
    ("Tier 0F - Gemma 3n E4B",        "google/gemma-3n-e4b-it",                    "NVIDIA_GEMMA_3N_API_KEY"),
    ("Tier 1  - Nemotron Nano 9B v2", "nvidia/nvidia-nemotron-nano-9b-v2",         "NVIDIA_NEMOTRON_NANO_9B_API_KEY"),
    ("Tier 2  - Nemotron Nano 30B",   "nvidia/nemotron-3-nano-30b-a3b",            "NVIDIA_NEMOTRON_NANO_30B_API_KEY"),
    ("Tier 3  - Nemotron Super 120B", "nvidia/nemotron-3-super-120b-a12b",         "NVIDIA_NEMOTRON_SUPER_API_KEY"),
    ("Tier 3F - Mistral Medium 3.5",  "mistralai/mistral-medium-3.5-128b",         "NVIDIA_MISTRAL_MEDIUM_API_KEY"),
    ("Tier 4  - Qwen3 Coder 480B",   "qwen/qwen3-coder-480b-a35b-instruct",       "NVIDIA_QWEN3_CODER_API_KEY"),
    ("Tier 4F - Qwen3.5 397B",        "qwen/qwen3.5-397b-a17b",                   "NVIDIA_QWEN3_5_API_KEY"),
    ("Optimizer - Nemotron Nano 30B", "nvidia/nemotron-3-nano-30b-a3b",            "NVIDIA_PROMPT_OPTIMIZER_API_KEY"),
]

async def test_model(client, label, model_id, key_env):
    key = os.getenv(key_env, "").strip()
    if not key:
        print(f"  [MISSING] {label:40s} | KEY NOT SET ({key_env})")
        return

    try:
        t0 = time.time()
        r = await client.post(
            NVIDIA_URL,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": TEST_PROMPT}],
                "max_tokens": 16,
                "temperature": 0.1,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        elapsed = round(time.time() - t0, 1)
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"][:60]
            print(f"  [OK]      {label:40s} | {elapsed}s | {text}")
        else:
            err = r.text[:150]
            print(f"  [FAIL]    {label:40s} | {elapsed}s | HTTP {r.status_code}: {err}")
    except Exception as e:
        print(f"  [ERROR]   {label:40s} | {str(e)[:120]}")

async def main():
    print("\nCORA API Diagnostics — New Model Stack")
    print("=" * 100)
    async with httpx.AsyncClient(timeout=60) as client:
        for label, model_id, key_env in MODELS:
            await test_model(client, label, model_id, key_env)
    print("=" * 100)

asyncio.run(main())
