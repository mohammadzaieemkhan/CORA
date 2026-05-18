"""
llm_providers.nemotron_nano_30b
───────────────────────────────
Tier 2 — NVIDIA Nemotron 3 Nano 30B-A3B
Mid-range MoE model with reasoning capabilities for
moderate analytical and coding queries.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : nvidia/nemotron-3-nano-30b-a3b
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"
DISPLAY_NAME = "Nemotron Nano 30B-A3B"
TIER = "Tier 2"
API_KEY_ENV = "NVIDIA_NEMOTRON_NANO_30B_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Nemotron Nano 30B-A3B and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=1.0,
        top_p=1.0,
        max_tokens=16384,
        extra_body={
            "reasoning_budget": 16384,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
