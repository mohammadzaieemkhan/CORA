"""
llm_providers.gemma_3n_e4b
──────────────────────────
Tier 0 (Fallback) — Google Gemma 3n E4B IT
Lightweight fallback model for simple queries.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : google/gemma-3n-e4b-it
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-3n-e4b-it"
DISPLAY_NAME = "Gemma 3n E4B"
TIER = "Tier 0"
API_KEY_ENV = "NVIDIA_GEMMA_3N_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Gemma 3n E4B and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.20,
        top_p=0.70,
        max_tokens=512,
    )
