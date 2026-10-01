"""Quick diagnostic: test each verified model endpoint in CORA across all tiers."""
import asyncio, os, sys, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import httpx

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TEST_PROMPT = "Say hello in one word."

MODELS = [
    # (Tier & Name, Provider, Model ID, Env Var)
    ("Tier 0 Primary  - Gemma 4 26B (a4b)",       "google",     "gemma-4-26b-a4b-it",                "GOOGLE_AI_STUDIO_API_KEY"),
    ("Tier 0 Fallback - Gemma 4 (OpenRouter)",    "openrouter", "google/gemma-4-26b-a4b-it:free",       "OPENROUTER_API_KEY"),
    ("Tier 1 Primary  - Gemini 3.5 Flash Lite",   "google",     "gemini-3.5-flash-lite",             "GOOGLE_AI_STUDIO_API_KEY"),
    ("Tier 1 Fallback - Llama 3.1 8B (OpenRouter)","openrouter","meta-llama/llama-3.1-8b-instruct:free", "OPENROUTER_API_KEY"),
    ("Tier 2 Primary  - Nemotron 3.5 Lightning",  "nvidia",     "nvidia/nemotron-3.5-lightning-30b-a3b", "NVIDIA_NEMOTRON_3_5_LIGHTNING_API_KEY"),
    ("Tier 2 Fallback - Muse Glimmer 30B",        "nvidia",     "meta/muse-glimmer-30b",              "NVIDIA_MUSE_GLIMMER_30B_API_KEY"),
    ("Tier 3 Primary  - Nemotron 3 Super 120B",   "nvidia",     "nvidia/nemotron-3-super-120b-a12b",  "NVIDIA_NEMOTRON_SUPER_API_KEY"),
    ("Tier 3 Fallback - GLM 5.3 Flash",           "nvidia",     "z-ai/glm-5.3-flash",                "NVIDIA_GLM_5_3_FLASH_API_KEY"),
    ("Tier 4 Primary  - Nemotron 3 Ultra 550B",   "nvidia",     "nvidia/nemotron-3-ultra-550b-a55b",  "NVIDIA_NEMOTRON_ULTRA_550B_API_KEY"),
    ("Tier 4 Fallback - Kimi K3",                 "nvidia",     "moonshotai/kimi-k3",                "NVIDIA_KIMI_K3_API_KEY"),
]

async def test_model(client, label, provider, model_id, key_env):
    key = os.getenv(key_env, "").strip()
    if not key:
        print(f"  [MISSING] {label:42s} | KEY NOT SET ({key_env})", flush=True)
        return

    try:
        t0 = time.time()
        if provider == "google":
            url = f"{GOOGLE_URL}/{model_id}:generateContent?key={key}"
            payload = {"contents": [{"parts": [{"text": TEST_PROMPT}]}]}
            r = await client.post(url, json=payload, timeout=25.0)
            elapsed = round(time.time() - t0, 1)
            if r.status_code == 200:
                data = r.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])
                text = parts[0].get("text", "").replace("\n", " ").strip()[:60]
                print(f"  [OK]      {label:42s} | {elapsed:4.1f}s | {text}", flush=True)
            else:
                err = r.text[:120]
                print(f"  [FAIL]    {label:42s} | {elapsed:4.1f}s | HTTP {r.status_code}: {err}", flush=True)
        elif provider == "openrouter":
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://cora.ai",
                "X-Title": "CORA",
            }
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": TEST_PROMPT}],
                "max_tokens": 32,
            }
            r = await client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=25.0)
            elapsed = round(time.time() - t0, 1)
            if r.status_code == 200:
                msg = r.json()["choices"][0]["message"]
                text = (msg.get("content") or msg.get("reasoning_content") or "").replace("\n", " ").strip()[:60]
                print(f"  [OK]      {label:42s} | {elapsed:4.1f}s | {text}", flush=True)
            else:
                err = r.text[:120]
                print(f"  [FAIL]    {label:42s} | {elapsed:4.1f}s | HTTP {r.status_code}: {err}", flush=True)
        else:
            r = await client.post(
                NVIDIA_URL,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": TEST_PROMPT}],
                    "max_tokens": 32,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=25.0,
            )
            elapsed = round(time.time() - t0, 1)
            if r.status_code == 200:
                msg = r.json()["choices"][0]["message"]
                text = (msg.get("content") or msg.get("reasoning_content") or "").replace("\n", " ").strip()[:60]
                print(f"  [OK]      {label:42s} | {elapsed:4.1f}s | {text}", flush=True)
            else:
                err = r.text[:120]
                print(f"  [FAIL]    {label:42s} | {elapsed:4.1f}s | HTTP {r.status_code}: {err}", flush=True)
    except Exception as e:
        print(f"  [ERROR]   {label:42s} | {str(e)[:120]}", flush=True)

async def main():
    print("\nCORA LLM Diagnostics — Active Verified Model Stack")
    print("=" * 110, flush=True)
    async with httpx.AsyncClient() as client:
        for label, provider, model_id, key_env in MODELS:
            await test_model(client, label, provider, model_id, key_env)
    print("=" * 110, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
