"""
llm_providers.nemotron_3_5_lightning_30b
────────────────────────────────────────
Tier 2 Primary — NVIDIA Nemotron 3.5 Lightning 30B-A3B
MoE reasoning model with extended thinking capabilities.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : nvidia/nemotron-3.5-lightning-30b-a3b
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/nemotron-3.5-lightning-30b-a3b"
DISPLAY_NAME = "Nemotron 3.5 Lightning 30B"
TIER = "Tier 2"
API_KEY_ENV = "NVIDIA_NEMOTRON_3_5_LIGHTNING_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Nemotron 3.5 Lightning 30B and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=1.0,
        top_p=0.95,
        max_tokens=8192,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 8192,
        },
        system_prompt=(
            "You are a helpful assistant. Be clear, accurate, and concise. "
            "Use markdown formatting where appropriate."
        ),
    )
