"""
llm_adapters.py
===============
Drop-in LLM clients for the pipeline. Each exposes .generate(prompt)->str.
Import the one matching your stack and pass it to VSARAG(llm=...).

These are optional; the pipeline defaults to MockLLM so it runs offline.
"""

from __future__ import annotations
import os
import re
import time


def _require_key(env_var: str, api_key: str | None) -> str:
    key = api_key or os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"{env_var} is not set. Export it or pass api_key=... explicitly."
        )
    return key


# Honors a server-specified delay like "Please retry in 54.1s" or
# "retryDelay': '54s'", so we wait the right amount on per-minute rate limits.
_RETRY_DELAY = re.compile(r"retry(?:delay)?[\"'\s:=]*?(\d+(?:\.\d+)?)\s*s", re.I)


def _retry(fn, attempts: int = 6, base: float = 1.5, max_wait: float = 65.0):
    """Call fn() with backoff on transient API errors.

    - Per-minute rate limits (Gemini/Groq free tiers) are honored by sleeping the
      server's stated retry delay, so the call eventually succeeds.
    - Per-DAY token caps cannot clear on a short retry, so we fail fast.
    - Re-raises the original error after the final attempt.
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            low = str(exc).lower()
            if "per day" in low or "tpd" in low or "perday" in low.replace(" ", ""):
                raise                       # daily cap — hopeless to retry
            if i == attempts - 1:
                raise
            m = _RETRY_DELAY.search(low)
            wait = min(float(m.group(1)) + 1.0, max_wait) if m else base ** (i + 1)
            time.sleep(wait)


class GroqLLM:
    """Matches a llama-3.1-8b-instant style Groq setup.

    temperature defaults to 0.2 for app use; pass temperature=0 for
    deterministic eval runs (removes run-to-run sampling variance).
    """
    def __init__(self, model: str = "llama-3.1-8b-instant",
                 api_key: str | None = None, temperature: float = 0.2):
        from groq import Groq  # pip install groq
        self.client = Groq(api_key=_require_key("GROQ_API_KEY", api_key))
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        r = _retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        ))
        return r.choices[0].message.content or ""


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None,
                 temperature: float = 0.2):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI(api_key=_require_key("OPENAI_API_KEY", api_key))
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        r = _retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        ))
        return r.choices[0].message.content or ""


class GeminiLLM:
    """Google Gemini via the google-genai SDK. Reads GEMINI_API_KEY.

    A class-level throttle spaces requests to respect the free tier's
    ~5 requests/minute limit so batch jobs (evals) don't get rate-limited out.
    """
    _last_call = 0.0
    _min_interval = 12.5   # seconds between calls (~5/min)

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None,
                 temperature: float = 0.2):
        from google import genai  # pip install google-genai
        self.client = genai.Client(api_key=_require_key("GEMINI_API_KEY", api_key))
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        gap = GeminiLLM._min_interval - (time.time() - GeminiLLM._last_call)
        if gap > 0:
            time.sleep(gap)
        try:
            r = _retry(lambda: self.client.models.generate_content(
                model=self.model, contents=prompt,
                config={"temperature": self.temperature}))
        finally:
            GeminiLLM._last_call = time.time()
        return (getattr(r, "text", None) or "")


class AnthropicLLM:
    # Default to Sonnet (balanced quality/cost); pass claude-haiku-4-5 for the
    # cheapest option or claude-opus-4-8 for max quality.
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None,
                 temperature: float = 0.2):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY", api_key))
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        r = _retry(lambda: self.client.messages.create(
            model=self.model, max_tokens=1024, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        ))
        return "".join(b.text for b in r.content if b.type == "text")
