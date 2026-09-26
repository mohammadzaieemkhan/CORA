"""
llm_providers.gemma_4_26b
─────────────────────────
Tier 0 (Primary) — Google Gemma 4 26B (a4b) IT
Ultra-lightweight 4B active parameter model for simple queries and tasks.

Provider : Google AI Studio (REST generateContent API)
Model ID : gemma-4-26b-a4b-it
"""

from __future__ import annotations

import os

from .base import call_gemini_rest

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "gemma-4-26b-a4b-it"
DISPLAY_NAME = "Gemma 4 26B (a4b)"
TIER = "Tier 0"
API_KEY_ENV = "GOOGLE_AI_STUDIO_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Google Gemma 4 26B (a4b) and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_gemini_rest(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.20,
        max_tokens=512,
        timeout=45.0,
        system_prompt=(
            "You are a helpful assistant. Be direct and concise. "
            "Answer in the fewest words necessary. Avoid filler, preambles, "
            "and unnecessary elaboration. Use markdown formatting when helpful."
        ),
    )
