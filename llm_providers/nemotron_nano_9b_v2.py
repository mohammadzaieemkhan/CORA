"""
llm_providers.nemotron_nano_9b_v2
─────────────────────────────────
Tier 1 — NVIDIA Nemotron Nano 9B v2
Mid-light model with built-in thinking/reasoning capabilities.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : nvidia/nvidia-nemotron-nano-9b-v2
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/nvidia-nemotron-nano-9b-v2"
DISPLAY_NAME = "Nemotron Nano 9B v2"
TIER = "Tier 1"
API_KEY_ENV = "NVIDIA_NEMOTRON_NANO_9B_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Nemotron Nano 9B v2 and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        extra_body={
            "min_thinking_tokens": 1024,
            "max_thinking_tokens": 2048,
        },
    )
