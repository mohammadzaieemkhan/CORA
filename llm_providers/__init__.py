"""
llm_providers
─────────────
Central registry for active LLM models used by CORA.

Working Model Stack:
  Tier 0 Primary   →  Google Gemma 4 26B (a4b)       (gemma-4-26b-a4b-it)
  Tier 0 Fallback  →  Gemma 4 26B (OpenRouter)       (google/gemma-4-26b-a4b-it:free)
  Tier 1 Primary   →  Google Gemini 3.5 Flash Lite   (gemini-3.5-flash-lite)
  Tier 1 Fallback  →  Google Gemma 2 9B (OpenRouter) (google/gemma-2-9b-it:free)
  Tier 2 Primary   →  Nemotron 3.5 Lightning 30B     (nvidia/nemotron-3.5-lightning-30b-a3b)
  Tier 2 Fallback  →  Meta Muse Glimmer 30B          (meta/muse-glimmer-30b)
  Tier 3 Primary   →  Nemotron 3 Super 120B          (nvidia/nemotron-3-super-120b-a12b)
  Tier 3 Fallback  →  DeepSeek V4.1 Flash (NVIDIA)   (deepseek-ai/deepseek-v4.1-flash)
  Tier 4 Primary   →  Nemotron 3 Ultra 550B          (nvidia/nemotron-3-ultra-550b-a55b)
  Tier 4 Fallback  →  DeepSeek V4.1 Flash / Kimi K3  (deepseek-ai/deepseek-v4.1-flash)

Prompt Optimizer:
  Google Gemini Top-Tier (gemini-3.8-flash / gemini-3.5-flash-lite)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from . import gemma_4_26b
from . import gemma_4_26b_openrouter
from . import gemini_3_5_flash_lite
from . import gemma_2_9b_openrouter
from . import nemotron_3_5_lightning_30b
from . import muse_glimmer_30b
from . import nemotron_super_120b
from . import glm_5_3_flash
from . import nemotron_3_ultra_550b
from . import kimi_k3
from . import deepseek_v4_1_flash
from .base import close_clients

logger = logging.getLogger("cora.llm")

# ── Model Registry (Active models) ──────────────────────────────────────────
MODEL_REGISTRY = [
    gemma_4_26b,                  # Tier 0 Primary
    gemma_4_26b_openrouter,       # Tier 0 Fallback
    gemini_3_5_flash_lite,        # Tier 1 Primary
    gemma_2_9b_openrouter,        # Tier 1 Fallback
    nemotron_3_5_lightning_30b,   # Tier 2 Primary
    muse_glimmer_30b,             # Tier 2 Fallback
    nemotron_super_120b,          # Tier 3 Primary
    glm_5_3_flash,                # Tier 3 Fallback
    nemotron_3_ultra_550b,        # Tier 4 Primary
    deepseek_v4_1_flash,          # Tier Fallback
    kimi_k3,                      # Tier 4 Fallback
]

# ── Tier → Model Mapping ────────────────────────────────────────────────────
TIER_MODEL_MAP = {
    "Tier 0": gemma_4_26b,
    "Tier 1": gemini_3_5_flash_lite,
    "Tier 2": nemotron_super_120b,
    "Tier 3": nemotron_super_120b,
    "Tier 4": nemotron_3_ultra_550b,
}

# ── Fallback chains per tier ────────────────────────────────────────────────
TIER_FALLBACKS = {
    "Tier 0": [gemini_3_5_flash_lite, nemotron_super_120b, gemma_4_26b_openrouter],
    "Tier 1": [gemma_4_26b, nemotron_super_120b, gemma_2_9b_openrouter],
    "Tier 2": [gemini_3_5_flash_lite, muse_glimmer_30b, nemotron_3_ultra_550b],
    "Tier 3": [nemotron_3_ultra_550b, kimi_k3, muse_glimmer_30b],
    "Tier 4": [kimi_k3, nemotron_super_120b, muse_glimmer_30b],
}


def _build_fallback_chain(primary_module):
    """Build a fallback chain: try every other working model in the registry."""
    return [m for m in MODEL_REGISTRY if m is not primary_module]


async def call_llm(
    tier: str,
    prompt: str,
    user_api_key: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Route a prompt to the correct model based on tier assignment.
    Falls back to tier-specific fallbacks first, then general fallback chain.

    Returns:
        (response_text, display_model_name)
    """
    primary = TIER_MODEL_MAP.get(tier, nemotron_3_5_lightning_30b)

    tier_fallbacks = TIER_FALLBACKS.get(tier, [])
    attempt_order = [primary] + tier_fallbacks + _build_fallback_chain(primary)

    # Deduplicate while preserving order
    seen = set()
    unique_order = []
    for m in attempt_order:
        if id(m) not in seen:
            seen.add(id(m))
            unique_order.append(m)

    errors = []
    for model_module in unique_order:
        try:
            key = user_api_key or model_module.get_api_key()
            if not key:
                logger.warning(f"No API key for {model_module.DISPLAY_NAME}, skipping.")
                continue

            logger.info(f"Calling {model_module.DISPLAY_NAME} ({model_module.MODEL_ID})")
            text = await model_module.call(prompt, key)
            return str(text or ""), model_module.DISPLAY_NAME

        except Exception as e:
            logger.error(f"LLM error ({model_module.DISPLAY_NAME}): {e}")
            errors.append(f"{model_module.DISPLAY_NAME}: {str(e)[:200]}")
            continue

    error_detail = " | ".join(errors) if errors else "No API keys configured"
    return f"[Error: {error_detail}]", primary.DISPLAY_NAME


def get_tier_model_info(tier: str) -> Tuple[str, str]:
    """Return (model_id, display_name) for the primary model of a tier."""
    module = TIER_MODEL_MAP.get(tier, nemotron_3_5_lightning_30b)
    return module.MODEL_ID, module.DISPLAY_NAME
