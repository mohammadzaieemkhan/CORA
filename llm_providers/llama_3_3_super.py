"""
llm_providers.llama_3_3_super
─────────────────────────────
Tier 3 — NVIDIA Llama 3.3 Nemotron Super 49B
High-performance reasoning model for complex logical tasks.
"""

from __future__ import annotations
import os
from .base import call_nvidia_openai

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/llama-3.3-nemotron-super-49b-v1"
DISPLAY_NAME = "Llama 3.3 Super (49B)"
TIER = "Tier 3"
API_KEY_ENV = "NVIDIA_LLAMA_SUPER_API_KEY"

def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()

async def call(prompt: str, api_key: str | None = None) -> str:
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV})")
    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.0,
        max_tokens=4096,
    )
