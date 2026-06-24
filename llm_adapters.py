"""
llm_adapters.py
===============
Drop-in LLM clients for the pipeline. Each exposes .generate(prompt)->str.
Import the one matching your stack and pass it to VSARAG(llm=...).

These are optional; the pipeline defaults to MockLLM so it runs offline.
"""

from __future__ import annotations
import os
import time


def _require_key(env_var: str, api_key: str | None) -> str:
    key = api_key or os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"{env_var} is not set. Export it or pass api_key=... explicitly."
        )
    return key


def _retry(fn, attempts: int = 3, base: float = 1.5):
    """Call fn() with exponential backoff on transient API errors.

    Fails fast on a daily/quota cap — a short retry cannot clear it — and
    re-raises the original error after the final attempt so callers still see it.
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            if any(s in msg for s in ("per day", "tpd", "quota", "insufficient")):
                raise
            if i == attempts - 1:
                raise
            time.sleep(base ** (i + 1))


class GroqLLM:
    """Matches a llama-3.1-8b-instant style Groq setup."""
    def __init__(self, model: str = "llama-3.1-8b-instant",
                 api_key: str | None = None):
        from groq import Groq  # pip install groq
        self.client = Groq(api_key=_require_key("GROQ_API_KEY", api_key))
        self.model = model

    def generate(self, prompt: str) -> str:
        r = _retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ))
        return r.choices[0].message.content or ""


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI(api_key=_require_key("OPENAI_API_KEY", api_key))
        self.model = model

    def generate(self, prompt: str) -> str:
        r = _retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ))
        return r.choices[0].message.content or ""


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY", api_key))
        self.model = model

    def generate(self, prompt: str) -> str:
        r = _retry(lambda: self.client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ))
        return "".join(b.text for b in r.content if b.type == "text")
