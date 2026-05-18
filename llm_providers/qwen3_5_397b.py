"""
llm_providers.qwen3_5_397b
──────────────────────────
Tier 4 (Fallback) — Qwen3.5 397B-A17B
Frontier MoE model as fallback for the most complex tasks.

Provider : NVIDIA Integrate API (OpenAI-compatible)
Model ID : qwen/qwen3.5-397b-a17b
"""

from __future__ import annotations

import os

from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "qwen/qwen3.5-397b-a17b"
DISPLAY_NAME = "Qwen3.5 397B"
TIER = "Tier 4"
API_KEY_ENV = "NVIDIA_QWEN3_5_API_KEY"


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Qwen3.5 397B-A17B and return the response text."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.60,
        top_p=0.95,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "top_k": 20,
            "presence_penalty": 0,
            "repetition_penalty": 1,
        },
    )
