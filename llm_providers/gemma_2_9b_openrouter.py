"""
llm_providers.gemma_2_9b_openrouter
───────────────────────────────────
Tier 1 Fallback: Google Gemma 2 9B (Free)
Provider: OpenRouter API with Google AI Studio backup

Model ID: google/gemma-2-9b-it:free
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from .base import call_openrouter, call_gemini_rest

logger = logging.getLogger("cora.llm.gemma_2_9b")

MODEL_ID = "google/gemma-2-9b-it:free"
DISPLAY_NAME = "Google Gemma 2 9B (Free)"
API_KEY_ENV = "OPENROUTER_API_KEY"
GOOGLE_BACKUP_ENV = "GOOGLE_AI_STUDIO_API_KEY"


def get_api_key() -> str:
    """Retrieve the OpenRouter API key for Gemma 2 9B."""
    return os.getenv(API_KEY_ENV, "").strip()


async def call(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Call google/gemma-2-9b-it:free via OpenRouter.
    If OpenRouter has no active endpoints (404), seamlessly falls back
    to Google AI Studio Gemma / Gemini using the Google API key.
    """
    key = api_key or get_api_key()
    
    # 1. Attempt OpenRouter with google/gemma-2-9b-it:free
    if key:
        try:
            logger.info(f"Calling OpenRouter with {MODEL_ID}...")
            return await call_openrouter(
                model=MODEL_ID,
                prompt=prompt,
                api_key=key,
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                timeout=15.0,
            )
        except Exception as e:
            logger.warning(f"OpenRouter {MODEL_ID} failed ({e}). Trying Google API key backup...")

    # 2. Resilient backup using Google AI Studio API key
    google_key = os.getenv(GOOGLE_BACKUP_ENV, "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key:
        try:
            logger.info("Calling Google AI Studio backup (gemini-3.5-flash-lite) for Tier 1...")
            return await call_gemini_rest(
                model="gemini-3.5-flash-lite",
                prompt=prompt,
                api_key=google_key,
                temperature=0.2,
                max_tokens=1024,
                timeout=15.0,
            )
        except Exception as ge:
            logger.warning(f"Google AI Studio backup failed: {ge}")

    # 3. Fallback to active OpenRouter free Gemma if available
    if key:
        try:
            return await call_openrouter(
                model="google/gemma-4-26b-a4b-it:free",
                prompt=prompt,
                api_key=key,
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                timeout=15.0,
            )
        except Exception as e3:
            raise Exception(f"All Tier 1 fallback endpoints failed: {e3}")

    raise Exception(f"No API key configured for {DISPLAY_NAME} ({API_KEY_ENV} or {GOOGLE_BACKUP_ENV})")
