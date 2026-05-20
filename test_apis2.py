"""Deeper diagnostic — tests each model with thinking params where applicable."""
import asyncio, os, time, traceback
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
import httpx

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TEST_PROMPT = "Say hello in one word."

MODELS = [
    # (label, model_id, key_env, extra_body)
    ("Nemotron Mini 4B",    "nvidia/nemotron-mini-4b-instruct",        "NVIDIA_NEMOTRON_MINI_API_KEY",      None),
    ("Gemma 3n E4B",        "google/gemma-3n-e4b-it",                  "NVIDIA_GEMMA_3N_API_KEY",           None),
    ("Nemotron Nano 9B v2", "nvidia/nvidia-nemotron-nano-9b-v2",       "NVIDIA_NEMOTRON_NANO_9B_API_KEY",   {"min_thinking_tokens": 128, "max_thinking_tokens": 512}),
    ("Nemotron Nano 30B",   "nvidia/nemotron-3-nano-30b-a3b",          "NVIDIA_NEMOTRON_NANO_30B_API_KEY",  {"reasoning_budget": 512, "chat_template_kwargs": {"enable_thinking": True}}),
    ("Nemotron Super 120B", "nvidia/nemotron-3-super-120b-a12b",       "NVIDIA_NEMOTRON_SUPER_API_KEY",     {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 512}),
    ("Mistral Medium 3.5",  "mistralai/mistral-medium-3.5-128b",       "NVIDIA_MISTRAL_MEDIUM_API_KEY",     {"reasoning_effort": "low"}),
    ("Qwen3 Coder 480B",    "qwen/qwen3-coder-480b-a35b-instruct",     "NVIDIA_QWEN3_CODER_API_KEY",        None),
    ("Qwen3.5 397B",        "qwen/qwen3.5-397b-a17b",                  "NVIDIA_QWEN3_5_API_KEY",            {"chat_template_kwargs": {"enable_thinking": True}}),
    ("Prompt Optimizer",    "nvidia/nemotron-3-nano-30b-a3b",          "NVIDIA_PROMPT_OPTIMIZER_API_KEY",   {"reasoning_budget": 512, "chat_template_kwargs": {"enable_thinking": True}}),
]

async def test(client, label, model_id, key_env, extra_body):
    key = os.getenv(key_env, "").strip()
    if not key:
        print(f"  [NO KEY]  {label:25s} | {key_env}")
        return
    try:
        t0 = time.time()
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "max_tokens": 32,
            "temperature": 0.1,
        }
        if extra_body:
            payload.update(extra_body)
        r = await client.post(
            NVIDIA_URL,
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        elapsed = round(time.time() - t0, 1)
        if r.status_code == 200:
            data = r.json()
            ch = data.get("choices", [{}])[0]
            msg = ch.get("message", {})
            text = msg.get("content", "") or ""
            print(f"  [OK]      {label:25s} | {elapsed}s | {text[:60]}")
        else:
            print(f"  [HTTP {r.status_code}] {label:25s} | {elapsed}s | {r.text[:200]}")
    except Exception as e:
        print(f"  [ERROR]   {label:25s} | {traceback.format_exc()[-200:]}")

async def main():
    print("\nCORA Deep Diagnostic — New Model Stack")
    print("=" * 100)
    async with httpx.AsyncClient(timeout=90) as c:
        for label, model_id, key_env, extra_body in MODELS:
            await test(c, label, model_id, key_env, extra_body)
    print("=" * 100)

asyncio.run(main())
