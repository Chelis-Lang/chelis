"""OpenRouter API client for chelis-tools."""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "qwen/qwen3.6-plus:free"

MODELS = {
    "qwen": "qwen/qwen3.6-plus:free",
    "trinity": "arcee-ai/trinity-large-preview:free",
    "nemotron": "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun": "stepfun/step-3.5-flash:free",
    "glm": "z-ai/glm-4.5-air:free",
}

# Fallback order when a model is rate-limited or unavailable
FALLBACK_CHAIN = [
    "qwen/qwen3.6-plus:free",
    "arcee-ai/trinity-large-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun/step-3.5-flash:free",
    "z-ai/glm-4.5-air:free",
]

RETRYABLE_CODES = {429, 502, 503}

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_api_key(env_path=None):
    """Load API key from env var or .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    candidates = [env_path] if env_path else []
    candidates.extend([".env", "../.env"])
    for path in candidates:
        if path is None:
            continue
        p = Path(path)
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "OPENROUTER_API_KEY not found in environment or .env file"
    )


def strip_think(text):
    """Strip <think> tags from reasoning model responses."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def _retry_after(headers):
    """Parse Retry-After header, return seconds to wait or None."""
    val = headers.get("Retry-After")
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _request_once(messages, model, api_key, max_tokens, temperature):
    """Single request attempt. Returns content string or raises."""
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    if "choices" not in body:
        err = body.get("error", {})
        raise RuntimeError(f"OpenRouter error: {err.get('message', body)}")
    return strip_think(body["choices"][0]["message"]["content"])


def _try_model(messages, model, api_key, max_tokens, temperature, max_retries=3):
    """Try a single model with retries on transient errors. Returns content or raises."""
    for attempt in range(max_retries + 1):
        try:
            return _request_once(messages, model, api_key, max_tokens, temperature)
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE_CODES or attempt == max_retries:
                raise
            wait = _retry_after(e.headers) or (2 ** attempt * 5)
            wait = min(wait, 60)
            print(f"  {e.code} from {model}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
            time.sleep(wait)


def chat(messages, model=DEFAULT_MODEL, api_key=None, max_tokens=4096, temperature=0, fallback=True):
    """Send a chat completion request to OpenRouter.

    If fallback=True and the requested model fails with a retryable error,
    automatically tries the remaining models in FALLBACK_CHAIN order.
    """
    if api_key is None:
        api_key = load_api_key()

    if fallback:
        # Build chain: requested model first, then remaining fallbacks in order
        chain = [model] + [m for m in FALLBACK_CHAIN if m != model]
    else:
        chain = [model]

    last_err = None
    for m in chain:
        try:
            print(f"Trying {m}...", file=sys.stderr)
            return _try_model(messages, m, api_key, max_tokens, temperature)
        except (urllib.error.HTTPError, RuntimeError) as e:
            last_err = e
            print(f"  {m} failed: {e}", file=sys.stderr)
            continue

    raise RuntimeError(f"All models exhausted. Last error: {last_err}")
