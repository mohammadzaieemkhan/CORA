"""
llm_providers.nemotron_3_ultra_550b
───────────────────────────────────
Tier 4 (Fallback) — NVIDIA Nemotron 3 Ultra 550B-A55B
Ultra-scale MoE reasoning frontier model for multi-step reasoning, complex code, and math.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : nvidia/nemotron-3-ultra-550b-a55b
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b"
DISPLAY_NAME = "Nemotron 3 Ultra 550B"
TIER = "Tier 4"
API_KEY_ENV = "NVIDIA_NEMOTRON_ULTRA_550B_API_KEY"


def get_api_key() -> str:
    return (
        os.getenv("NVIDIA_NEMOTRON_ULTRA_550B_API_KEY", "")
        or os.getenv("NVIDIA_QWEN3_5_API_KEY", "")
    ).strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Nemotron 3 Ultra 550B and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
        timeout=25.0,
        system_prompt=(
            "You are a highly capable assistant for complex reasoning tasks. "
            "Provide comprehensive, well-structured, and accurate answers. "
            "Use markdown headings, lists, and code blocks where helpful."
        ),
    )
