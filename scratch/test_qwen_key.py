import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
key = os.getenv("NVIDIA_QWEN3_5_API_KEY", "").strip()

url = "https://integrate.api.nvidia.com/v1/chat/completions"

print(f"Testing key: {key[:15]}...")

# Test with a lightweight model that always works
r = requests.post(
    url,
    json={
        "model": "google/gemma-3n-e4b-it",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    },
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    timeout=10
)

print(f"HTTP Status: {r.status_code}")
print(f"Response: {r.text[:300]}")
