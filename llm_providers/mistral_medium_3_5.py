"""
llm_providers.mistral_medium_3_5
────────────────────────────────
Tier 3 (Fallback) — Mistral Medium 3.5 128B
Strong reasoning model as fallback for complex queries.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : mistralai/mistral-medium-3.5-128b
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/mistral-medium-3.5-128b"
DISPLAY_NAME = "Mistral Medium 3.5"
TIER = "Tier 3"
API_KEY_ENV = "NVIDIA_MISTRAL_MEDIUM_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Mistral Medium 3.5 128B and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.70,
        top_p=1.0,
        max_tokens=16384,
        extra_body={
            "reasoning_effort": "high",
        },
    )
