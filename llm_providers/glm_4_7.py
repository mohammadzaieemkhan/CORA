"""
llm_providers.glm_4_7
─────────────────────
Tier 1 — Z-AI GLM 4.7 (9B)
Moderate model for balanced analytical / factual queries.
"""

from __future__ import annotations
import os
from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "z-ai/glm4.7"
DISPLAY_NAME = "GLM 4.7 (9B)"
TIER = "Tier 1"
API_KEY_ENV = "NVIDIA_GLM_API_KEY"

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
        temperature=0.3,
        max_tokens=1024,
    )
