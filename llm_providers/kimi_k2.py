"""
llm_providers.kimi_k2
─────────────────────
Tier 4 — Moonshot AI Kimi K2 Thinking
Frontier reasoning model for the most complex multi-step problems.
"""

from __future__ import annotations
import os
from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "moonshotai/kimi-k2-thinking"
DISPLAY_NAME = "Kimi K2 Thinking"
TIER = "Tier 4"
API_KEY_ENV = "NVIDIA_KIMI_API_KEY"

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
        temperature=0.0,
        max_tokens=8192,
    )
