"""
llm_providers.glm_5_3_flash
───────────────────────────
Tier 3 (Fallback) — Zhipu AI GLM 5.3 Flash
High-throughput reasoning model hosted on NVIDIA Integrate API.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : z-ai/glm-5.3-flash
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "z-ai/glm-5.3-flash"
DISPLAY_NAME = "GLM 5.3 Flash"
TIER = "Tier 3"
API_KEY_ENV = "NVIDIA_GLM_5_3_FLASH_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to GLM 5.3 Flash and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.5,
        top_p=1.0,
        max_tokens=1024,
        timeout=20,
        system_prompt="You are a helpful assistant.",
    )
