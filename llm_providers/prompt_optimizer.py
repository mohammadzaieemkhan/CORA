"""
llm_providers.prompt_optimizer
──────────────────────────────
Dedicated model for Prompt Optimization: Mistral Large 3 675B
Provider : NVIDIA Integrate API

Uses Mistral Large 3 as the optimizer engine (Devstral 2 123B is
currently DEGRADED on NVIDIA).

OPTIMISED: Persistent client, reduced max_tokens, tighter timeout.
"""

from __future__ import annotations

import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger("cora.llm.optimizer")

MODEL_ID = "mistralai/mistral-large-3-675b-instruct-2512"
API_KEY_ENV = "NVIDIA_MISTRAL_LARGE_API_KEY"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Persistent client ────────────────────────────────────────────────────────
_client: AsyncOpenAI | None = None


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


def _get_client(key: str) -> AsyncOpenAI:
    """Return a persistent AsyncOpenAI client for the optimizer."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=key,
            timeout=30.0,
            max_retries=0,
        )
    return _client


async def optimize_prompt(prompt: str, api_key: str | None = None) -> str:
    """Use Mistral Large 3 to automatically restructure a prompt for maximum clarity."""
    key = api_key or get_api_key()
    if not key:
        raise Exception(f"No API key configured for Prompt Optimizer ({API_KEY_ENV})")
        
    system_instruction = (
        "You are an expert prompt compiler. The user is submitting a prompt to an AI logic router. "
        "Your task is strictly to COMPRESS and OPTIMIZE their raw prompt for maximum token efficiency and clarity.\n"
        "RULES:\n"
        "1. Extract ONLY the core actionable intent or question.\n"
        "2. If a large portion of the prompt is irrelevant rambling, preamble, or context that does not affect the final question, COMPLETELY DISCARD IT.\n"
        "3. DO NOT add any new logic, constraints, features, or 'technical requirements' that the user did not explicitly state.\n"
        "4. Use concise, direct language. Remove all conversational filler.\n"
        "5. DO NOT ANSWER THEIR QUESTION. Return strictly the optimized, compressed prompt text ready for execution.\n\n"
        f"RAW PROMPT TO OPTIMIZE:\n{prompt}"
    )
    
    client = _get_client(key)
    
    full_content = ""

    completion = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": system_instruction}],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stream=True,
    )

    async for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
            
        delta = chunk.choices[0].delta
        
        # Check for reasoning_content just in case this model supports it
        reasoning = getattr(delta, "reasoning_content", None)
        # We drop reasoning explicitly for optimization since we only want the cleanly formatted prompt
        
        if delta.content is not None:
            full_content += delta.content

    return full_content.strip()
