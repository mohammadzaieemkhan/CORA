"""
llm_providers.base
───────────────────
Shared constants and async HTTP helpers for LLM provider calls.
All NVIDIA models use the OpenAI-compatible chat/completions endpoint.

OPTIMISED: Uses a persistent httpx.AsyncClient pool instead of creating
a new client per request. Eliminates ~200-500ms of TCP/TLS overhead.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger("cora.llm")

# ── NVIDIA Integrate API ─────────────────────────────────────────────────────
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_ENDPOINT = f"{NVIDIA_BASE_URL}/chat/completions"

# ── GitHub Models API (for DeepSeek V3) ──────────────────────────────────────
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
GITHUB_CHAT_ENDPOINT = f"{GITHUB_MODELS_BASE_URL}/chat/completions"

# ── Default generation params ────────────────────────────────────────────────
DEFAULT_TIMEOUT = 30  # Reduced from 60s — fail fast, don't hang

# ── Persistent client pool ───────────────────────────────────────────────────
# One shared client per base URL. Reuses TCP connections across requests.
_client_pool: dict[str, httpx.AsyncClient] = {}


def _get_client(base_url: str = NVIDIA_CHAT_ENDPOINT) -> httpx.AsyncClient:
    """Get or create a persistent async HTTP client for the given base URL."""
    if base_url not in _client_pool:
        _client_pool[base_url] = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            # http2=True requires the 'h2' package; disabled — HTTP/1.1 works
            # fine with NVIDIA's API and the connection pool still reuses TCP.
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=120,
            ),
        )
    return _client_pool[base_url]


async def close_clients():
    """Shutdown all persistent clients (call at app shutdown)."""
    for client in _client_pool.values():
        await client.aclose()
    _client_pool.clear()


async def call_nvidia_openai(
    model: str,
    prompt: str,
    api_key: str,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 1024,
    extra_body: Optional[dict] = None,
) -> str:
    """
    Call an NVIDIA-hosted model via the OpenAI-compatible chat/completions API.
    Used by: Nemotron Mini/Nano/Super, Gemma 3n, Mistral Medium 3.5, Qwen3 Coder,
    Qwen3.5, and other reasoning models.

    Thinking-model note: models with reasoning/thinking enabled (e.g. Nemotron Nano
    9B v2, Nemotron Nano 30B, Nemotron Super 120B) may return `content=None` and
    put their final answer in `reasoning_content`. We fall back to `reasoning_content`
    so the caller always receives a non-None string.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)

    client = _get_client(NVIDIA_CHAT_ENDPOINT)
    r = await client.post(NVIDIA_CHAT_ENDPOINT, json=payload, headers=headers)
    if r.status_code != 200:
        raise Exception(f"NVIDIA API {r.status_code}: {r.text[:500]}")

    # Thinking models (enable_thinking=True / min_thinking_tokens) may return
    # content=None with the answer in reasoning_content.  Always return a string.
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content")
    if content is not None:
        return content
    # Fall back to reasoning_content (thinking models)
    reasoning = msg.get("reasoning_content") or ""
    return reasoning


async def call_github_openai(
    model: str,
    prompt: str,
    api_key: str,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Call a GitHub Models-hosted model via the OpenAI-compatible chat/completions API.
    Used by: DeepSeek V3.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    client = _get_client(GITHUB_CHAT_ENDPOINT)
    r = await client.post(GITHUB_CHAT_ENDPOINT, json=payload, headers=headers)
    if r.status_code != 200:
        raise Exception(f"GitHub Models API {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]["content"]


async def call_gemini_rest(
    model: str,
    prompt: str,
    api_key: str,
) -> str:
    """
    Call Google Gemini via the REST generateContent endpoint.
    Used by: Gemini 2.5 Flash.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    client = _get_client("https://generativelanguage.googleapis.com")
    r = await client.post(url, json=payload)
    if r.status_code != 200:
        raise Exception(f"Gemini API {r.status_code}: {r.text[:500]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
