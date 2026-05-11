"""
llm_providers.glm_5_1
─────────────────────
Tier 2 — Z-AI GLM 5.1 (9B)
Optimized mid-tier model for reasoning and complex analysis.
"""

from __future__ import annotations
import os
from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "z-ai/glm-5.1"
DISPLAY_NAME = "GLM 5.1 (9B)"
TIER = "Tier 2"
API_KEY_ENV = "NVIDIA_GLM_51_API_KEY"

def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()

async def call(prompt: str, api_key: str | None = None) -> str:
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.2,
        max_tokens=2048,
    )
