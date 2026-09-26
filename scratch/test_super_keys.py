import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

keys = {
    "NVIDIA_NEMOTRON_SUPER_API_KEY": os.getenv("NVIDIA_NEMOTRON_SUPER_API_KEY"),
    "NVIDIA_NEMOTRON_MINI_API_KEY": os.getenv("NVIDIA_NEMOTRON_MINI_API_KEY"),
    "NVIDIA_GEMMA_3N_API_KEY": os.getenv("NVIDIA_GEMMA_3N_API_KEY"),
    "NVIDIA_NEMOTRON_NANO_9B_API_KEY": os.getenv("NVIDIA_NEMOTRON_NANO_9B_API_KEY"),
    "NVIDIA_NEMOTRON_NANO_30B_API_KEY": os.getenv("NVIDIA_NEMOTRON_NANO_30B_API_KEY"),
    "NVIDIA_PROMPT_OPTIMIZER_API_KEY": os.getenv("NVIDIA_PROMPT_OPTIMIZER_API_KEY"),
}

url = "https://integrate.api.nvidia.com/v1/chat/completions"
model_id = "nvidia/nemotron-3-super-120b-a12b"

for name, key in keys.items():
    if not key:
        print(f"{name:35s}: KEY NOT FOUND in env")
        continue
    
    key = key.strip()
    try:
        r = requests.post(
            url,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 16,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=10
        )
        if r.status_code == 200:
            print(f"{name:35s}: SUCCESS (HTTP 200) - {r.json()['choices'][0]['message']['content'].strip()}")
        else:
            print(f"{name:35s}: FAILED (HTTP {r.status_code}) - {r.text[:150]}")
    except Exception as e:
        print(f"{name:35s}: ERROR - {e}")
