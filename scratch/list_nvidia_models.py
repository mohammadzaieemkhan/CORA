import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
key = os.getenv("NVIDIA_NEMOTRON_SUPER_API_KEY", "").strip()

url = "https://integrate.api.nvidia.com/v1/models"
headers = {
    "Authorization": f"Bearer {key}",
    "Accept": "application/json"
}

r = requests.get(url, headers=headers)
if r.status_code == 200:
    models = r.json().get("data", [])
    print(f"Total models available: {len(models)}")
    # Find models with "nemotron" in the ID
    nemotron_models = [m["id"] for m in models if "nemotron" in m["id"].lower()]
    print("\nNemotron models:")
    for m in sorted(nemotron_models):
        print(f"  {m}")
        
    print("\nAll models:")
    for m in sorted([m["id"] for m in models]):
        if "nemotron" not in m.lower():
            print(f"  {m}")
else:
    print(f"Failed: HTTP {r.status_code} - {r.text}")
