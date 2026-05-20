"""Quick test for Qwen3 Coder 480B (replaces Kimi K2 at Tier 4)."""
import asyncio, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
import httpx

async def main():
    key = os.getenv("NVIDIA_QWEN3_CODER_API_KEY", "").strip()
    if not key:
        print("NVIDIA_QWEN3_CODER_API_KEY not set")
        return
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            json={
                "model": "qwen/qwen3-coder-480b-a35b-instruct",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 16,
                "temperature": 0.1,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Response:", r.json()["choices"][0]["message"]["content"])
        else:
            print("Error:", r.text[:300])

asyncio.run(main())
