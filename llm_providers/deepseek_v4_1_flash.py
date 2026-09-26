"""
llm_providers.deepseek_v4_1_flash
─────────────────────────────────
Tier Fallback: DeepSeek V4.1 Flash
Provider: NVIDIA Integrate API (https://integrate.api.nvidia.com/v1)

Model ID: deepseek-ai/deepseek-v4.1-flash
A high-efficiency multimodal MoE model deployed on NVIDIA Integrate.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from .base import call_nvidia_openai

logger = logging.getLogger("cora.llm.deepseek_v4_1_flash")

MODEL_ID = "deepseek-ai/deepseek-v4.1-flash"
DISPLAY_NAME = "DeepSeek V4.1 Flash"
API_KEY_ENV = "NVIDIA_DEEPSEEK_V4_1_FLASH_API_KEY"


def get_api_key() -> str:
    """Retrieve the API key for DeepSeek V4.1 Flash."""
    return (
        os.getenv(API_KEY_ENV, "").strip()
        or os.getenv("NVIDIA_API_KEY", "").strip()
        or os.getenv("NVIDIA_NEMOTRON_SUPER_API_KEY", "").strip()
        or os.getenv("NVIDIA_MUSE_GLIMMER_30B_API_KEY", "").strip()
    )


async def call(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Call deepseek-ai/deepseek-v4.1-flash via the NVIDIA OpenAI-compatible API.
    """
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV} / NVIDIA_API_KEY)")

    return await call_nvidia_openai(
        model=MODEL_ID,
        prompt=prompt,
        api_key=key,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        timeout=30.0,
    )
