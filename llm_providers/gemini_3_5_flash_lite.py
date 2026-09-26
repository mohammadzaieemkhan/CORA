"""
llm_providers.gemini_3_5_flash_lite
───────────────────────────────────
Tier 1 (Primary) — Google Gemini 3.5 Flash Lite
Ultra-fast, efficient lightweight model for structured reasoning and medium tasks.

Provider : Google AI Studio (REST generateContent API)
Model ID : gemini-3.5-flash-lite
"""

from __future__ import annotations

import os

from .base import call_gemini_rest

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "gemini-3.5-flash-lite"
DISPLAY_NAME = "Gemini 3.5 Flash Lite"
TIER = "Tier 1"
API_KEY_ENV = "GOOGLE_AI_STUDIO_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Google Gemini 3.5 Flash Lite and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_gemini_rest(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.20,
        max_tokens=1024,
        timeout=30.0,
        system_prompt=(
            "You are a helpful assistant. Be direct and concise. "
            "Answer in the fewest words necessary. Avoid filler, preambles, "
            "and unnecessary elaboration. Use markdown formatting when helpful."
        ),
    )
