"""
llm_providers.gemma_4_26b_openrouter
─────────────────────────────────────
Tier 0 (Fallback) — Google Gemma 4 26B (a4b) IT (OpenRouter Free)
Fallback routing endpoint via OpenRouter.

Provider : OpenRouter (OpenAI-compatible)
Model ID : google/gemma-4-26b-a4b-it:free
"""

from __future__ import annotations

import os

from .base import call_openrouter

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-4-26b-a4b-it:free"
DISPLAY_NAME = "Gemma 4 26B (OpenRouter)"
TIER = "Tier 0"
API_KEY_ENV = "OPENROUTER_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Gemma 4 26B via OpenRouter and return response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_openrouter(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.20,
        max_tokens=512,
        timeout=30.0,
        system_prompt=(
            "You are a helpful assistant. Be direct and concise. "
            "Answer in the fewest words necessary. Avoid filler, preambles, "
            "and unnecessary elaboration. Use markdown formatting when helpful."
        ),
    )
