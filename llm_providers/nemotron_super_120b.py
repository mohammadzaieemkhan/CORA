"""
llm_providers.nemotron_super_120b
─────────────────────────────────
Tier 3 — NVIDIA Nemotron 3 Super 120B-A12B
High-performance MoE model for complex reasoning with
thinking capabilities.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : nvidia/nemotron-3-super-120b-a12b
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"
DISPLAY_NAME = "Nemotron 3 Super 120B"
TIER = "Tier 3"
API_KEY_ENV = "NVIDIA_NEMOTRON_SUPER_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Nemotron 3 Super 120B and return the response text."""
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
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 16384,
        },
    )
