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
import asyncio
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

# ── OpenRouter API ───────────────────────────────────────────────────────────
OPENROUTER_CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# ── Default generation params ────────────────────────────────────────────────
DEFAULT_TIMEOUT = 120  # 120s to accommodate heavy MoE / thinking model latency

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
    system_prompt: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """
    Call an NVIDIA-hosted model via the OpenAI-compatible chat/completions API.
    Used by: Nemotron Mini/Nano/Super, Gemma 3n, Mistral Medium 3.5, Qwen3 Coder,
    Qwen3.5, Kimi K3, and other reasoning models.

    Thinking-model note: models with reasoning/thinking enabled (e.g. Nemotron Nano
    9B v2, Nemotron Nano 30B, Nemotron Super 120B) may return `content=None` and
    put their final answer in `reasoning_content`. We fall back to `reasoning_content`
    so the caller always receives a non-None string.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if extra_body:
        payload.update(extra_body)

    client = _get_client(NVIDIA_CHAT_ENDPOINT)
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            kwargs = {"json": payload, "headers": headers}
            if timeout is not None:
                kwargs["timeout"] = timeout
            r = await client.post(NVIDIA_CHAT_ENDPOINT, **kwargs)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                logger.warning(f"Transient HTTP {r.status_code} for {model}. Retrying in {0.5 * (attempt + 1)}s...")
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise Exception(f"NVIDIA API {r.status_code}: {r.text[:500]}")
        except httpx.TimeoutException:
            logger.warning(f"Timeout calling {model}; falling back to next model.")
            raise
        except httpx.ConnectError as e:
            if attempt < max_retries:
                logger.warning(f"Transient connection error for {model}: {e}. Retrying in {0.5 * (attempt + 1)}s...")
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise

    # Thinking models (enable_thinking=True / min_thinking_tokens) may return
    # content=None with the answer in reasoning_content.  Always return a string.
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content")
    if content is not None and content.strip():
        return content
    # Fall back to reasoning_content (thinking models)
    reasoning = msg.get("reasoning_content") or ""
    return reasoning or (content or "")


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
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> str:
    """
    Call Google Gemini / Gemma via the REST generateContent endpoint.
    Used by: Google AI Studio models (Gemma 4, Gemini Flash Lite, etc.).
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload: dict = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    
    gen_config: dict = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = max_tokens
    if gen_config:
        payload["generationConfig"] = gen_config

    client = _get_client("https://generativelanguage.googleapis.com")
    kwargs = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        r = await client.post(url, json=payload, **kwargs)
    except (httpx.TimeoutException, TimeoutError) as e:
        logger.warning(f"Google AI Studio request timed out for {model}: {e}")
        raise

    if r.status_code != 200:
        raise Exception(f"Google AI Studio API {r.status_code}: {r.text[:500]}")
    
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        feedback = data.get("promptFeedback", {})
        raise Exception(f"Google AI Studio API returned no candidates. Feedback: {feedback}")
    
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        finish_reason = candidates[0].get("finishReason", "UNKNOWN")
        raise Exception(f"Google AI Studio API returned empty parts. Finish reason: {finish_reason}")
    
    return parts[0].get("text", "")


async def call_openrouter(
    model: str,
    prompt: str,
    api_key: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 1024,
    timeout: Optional[float] = 30.0,
) -> str:
    """
    Call an OpenRouter-hosted model via the OpenAI-compatible chat/completions API.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cora.ai",
        "X-Title": "CORA",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    client = _get_client(OPENROUTER_CHAT_ENDPOINT)
    kwargs = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        r = await client.post(OPENROUTER_CHAT_ENDPOINT, json=payload, headers=headers, **kwargs)
    except (httpx.TimeoutException, TimeoutError) as e:
        logger.warning(f"OpenRouter request timed out for {model}: {e}")
        raise

    if r.status_code != 200:
        raise Exception(f"OpenRouter API {r.status_code}: {r.text[:500]}")

    data = r.json()
    choices = data.get("choices", [])
    if not choices:
        raise Exception(f"OpenRouter returned no choices: {data}")
    msg = choices[0].get("message", {})
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return str(content)


