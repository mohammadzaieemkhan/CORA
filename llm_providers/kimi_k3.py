"""
llm_providers.kimi_k3
─────────────────────
Tier 4 — Moonshot AI Kimi K3
Frontier reasoning model hosted on NVIDIA Integrate API.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : moonshotai/kimi-k3
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "moonshotai/kimi-k3"
DISPLAY_NAME = "Kimi K3"
TIER = "Tier 4"
API_KEY_ENV = "NVIDIA_KIMI_K3_API_KEY"


def get_api_key() -> str:
    return (
        os.getenv("NVIDIA_KIMI_K3_API_KEY", "")
        or os.getenv("NVIDIA_QWEN3_CODER_API_KEY", "")
    ).strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Kimi K3 and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=1.0,
        top_p=0.95,
        max_tokens=16384,
        extra_body={
            "reasoning_effort": "max",
            "seed": 0,
        },
        timeout=20,
        system_prompt=(
            "You are a highly capable frontier assistant for complex reasoning and code. "
            "Provide comprehensive, structured, and accurate answers using markdown."
        ),
    )
