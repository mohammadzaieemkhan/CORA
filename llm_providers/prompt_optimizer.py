"""
llm_providers.prompt_optimizer
──────────────────────────────
Dedicated model for Prompt Optimization: Google Gemini Top-Tier
Provider : Google AI Studio

Uses Google's top-tier Gemini model (gemini-3.8-flash with fallback to
gemini-3.5-flash-lite) to compress and optimize raw user prompts for
maximum token efficiency and execution clarity.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from .base import call_gemini_rest

logger = logging.getLogger("cora.llm.optimizer")

PRIMARY_MODEL_ID = "gemini-3.5-flash-lite"
FALLBACK_MODEL_ID = "gemini-3.8-flash"
MODEL_ID = PRIMARY_MODEL_ID
DISPLAY_NAME = "Google Gemini 3.5 Flash Lite (Optimizer)"
API_KEY_ENV = "GOOGLE_AI_STUDIO_API_KEY"


def get_api_key() -> str:
    """Retrieve the Google AI Studio API key."""
    return (
        os.getenv(API_KEY_ENV, "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )


async def optimize_prompt(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Use Google Gemini to automatically restructure and compress a prompt for maximum clarity.
    Extracts the core actionable intent while discarding rambling filler and context.
    """
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for Prompt Optimizer ({API_KEY_ENV})")

    system_instruction = (
        "You are an expert prompt compiler and optimizer. The user is submitting a prompt to an AI logic router.\n"
        "Your task is strictly to COMPRESS and OPTIMIZE their raw prompt for maximum token efficiency and clarity.\n"
        "RULES:\n"
        "1. Extract ONLY the core actionable intent or question.\n"
        "2. If a large portion of the prompt is irrelevant rambling, preamble, or context that does not affect the final question, COMPLETELY DISCARD IT.\n"
        "3. DO NOT add any new logic, constraints, features, or 'technical requirements' that the user did not explicitly state.\n"
        "4. Use concise, direct language. Remove all conversational filler.\n"
        "5. DO NOT ANSWER THEIR QUESTION. Return strictly the optimized, compressed prompt text ready for execution."
    )

    user_content = f"RAW PROMPT TO OPTIMIZE:\n{prompt}"

    # Try Primary Top-Tier Model first
    try:
        logger.info(f"Optimizing prompt with Google {PRIMARY_MODEL_ID}...")
        result = await call_gemini_rest(
            model=PRIMARY_MODEL_ID,
            prompt=user_content,
            system_prompt=system_instruction,
            api_key=key,
            temperature=0.2,
            max_tokens=2048,
            timeout=15.0,
        )
        if result and result.strip():
            return result.strip()
    except Exception as e:
        logger.warning(f"Primary optimizer {PRIMARY_MODEL_ID} failed ({e}), falling back to {FALLBACK_MODEL_ID}...")

    # Fallback to Gemini 3.5 Flash Lite
    try:
        logger.info(f"Optimizing prompt with Google {FALLBACK_MODEL_ID}...")
        result = await call_gemini_rest(
            model=FALLBACK_MODEL_ID,
            prompt=user_content,
            system_prompt=system_instruction,
            api_key=key,
            temperature=0.2,
            max_tokens=2048,
            timeout=15.0,
        )
        if result and result.strip():
            return result.strip()
    except Exception as fb_err:
        logger.error(f"Fallback optimizer {FALLBACK_MODEL_ID} failed: {fb_err}")
        err_str = str(fb_err).lower()
        if "timed out" in err_str or "timeout" in err_str:
            raise Exception(
                "Prompt optimization timed out — the Google API is busy right now. "
                "Please try again in a moment, or submit your prompt directly without optimization."
            ) from fb_err
        raise

    return prompt.strip()
