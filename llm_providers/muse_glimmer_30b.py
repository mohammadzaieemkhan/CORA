"""
llm_providers.muse_glimmer_30b
───────────────────────────────
Tier 2 Fallback — Meta Muse Glimmer 30B
Mid-range reasoning and analytical model hosted on NVIDIA Integrate API.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : meta/muse-glimmer-30b
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "meta/muse-glimmer-30b"
DISPLAY_NAME = "Muse Glimmer 30B"
TIER = "Tier 2"
API_KEY_ENV = "NVIDIA_MUSE_GLIMMER_30B_API_KEY"


def get_api_key() -> str:
    return (
        os.getenv("NVIDIA_MUSE_GLIMMER_30B_API_KEY", "")
        or os.getenv("NVIDIA_NEMOTRON_NANO_30B_API_KEY", "")
    ).strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Meta Muse Glimmer 30B and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.2,
        top_p=0.9,
        max_tokens=2048,
        timeout=25.0,
        system_prompt=(
            "You are a helpful assistant. Be clear and well-structured. "
            "Provide thorough but concise answers. Avoid unnecessary repetition "
            "or excessive preambles. Use markdown formatting for clarity."
        ),
    )
