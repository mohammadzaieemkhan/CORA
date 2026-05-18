"""
llm_providers
─────────────
Central registry for all LLM models used by CORA.

Each model lives in its own file with a consistent interface:
  - MODEL_ID, DISPLAY_NAME, TIER, get_api_key(), call(prompt, api_key?)

This __init__ exports:
  - TIER_MODEL_MAP   — maps tier names to (module, display_name) tuples
  - call_llm()       — routes a prompt to the right model with fallback
  - MODEL_REGISTRY   — ordered list of all registered model modules

Tier Assignments (2026-05 refresh):
  Tier 0  →  Nemotron Mini 4B        (lightest, simple queries)
             Gemma 3n E4B            (fallback)
  Tier 1  →  Nemotron Nano 9B v2     (moderate factual / logical reasoning)
  Tier 2  →  Nemotron Nano 30B-A3B   (mid-range analytical / creative)
  Tier 3  →  Nemotron 3 Super 120B   (complex reasoning)
             Mistral Medium 3.5      (fallback)
  Tier 4  →  Qwen3 Coder 480B       (hardest multi-step / code)
             Qwen3.5 397B            (fallback)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from . import nemotron_mini_4b
from . import gemma_3n_e4b
from . import nemotron_nano_9b_v2
from . import nemotron_nano_30b
from . import nemotron_super_120b
from . import mistral_medium_3_5
from . import qwen3_coder
from . import qwen3_5_397b
from .base import close_clients

logger = logging.getLogger("cora.llm")

# ── Model Registry (Optimized for Stability) ────────────────────────────────
MODEL_REGISTRY = [
    nemotron_mini_4b,        # Tier 0
    gemma_3n_e4b,            # Tier 0 fallback
    nemotron_nano_9b_v2,     # Tier 1
    nemotron_nano_30b,       # Tier 2
    nemotron_super_120b,     # Tier 3
    mistral_medium_3_5,      # Tier 3 fallback
    qwen3_coder,             # Tier 4
    qwen3_5_397b,            # Tier 4 fallback
]

# ── Tier → Model Mapping ────────────────────────────────────────────────────
TIER_MODEL_MAP = {
    "Tier 0": nemotron_mini_4b,
    "Tier 1": nemotron_nano_9b_v2,
    "Tier 2": nemotron_nano_30b,
    "Tier 3": nemotron_super_120b,
    "Tier 4": qwen3_coder,
}

# ── Fallback chains per tier (if the primary model fails) ────────────────────
TIER_FALLBACKS = {
    "Tier 0": [gemma_3n_e4b],
    "Tier 1": [],
    "Tier 2": [],
    "Tier 3": [mistral_medium_3_5],
    "Tier 4": [qwen3_5_397b],
}


def _build_fallback_chain(primary_module):
    """Build a fallback chain: try every other model in the registry."""
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
    primary = TIER_MODEL_MAP.get(tier)
    if not primary:
        primary = nemotron_mini_4b  # ultimate fallback

    # Build attempt order: primary first, then tier-specific fallbacks,
    # then general fallback chain for robustness
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


# ── Convenience: get display info for a tier ─────────────────────────────────
def get_tier_model_info(tier: str) -> Tuple[str, str]:
    """Return (model_id, display_name) for the primary model of a tier."""
    module = TIER_MODEL_MAP.get(tier, nemotron_mini_4b)
    return module.MODEL_ID, module.DISPLAY_NAME
