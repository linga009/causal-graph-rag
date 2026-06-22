"""
llm_adapters.py
===============
Drop-in LLM clients for the pipeline. Each exposes .generate(prompt)->str.
Import the one matching your stack and pass it to VSARAG(llm=...).

These are optional; the pipeline defaults to MockLLM so it runs offline.
"""

from __future__ import annotations
import os


class GroqLLM:
    """Matches a llama-3.1-8b-instant style Groq setup."""
    def __init__(self, model: str = "llama-3.1-8b-instant",
                 api_key: str | None = None):
        from groq import Groq  # pip install groq
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model = model

    def generate(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return r.choices[0].message.content


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    def generate(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return r.choices[0].message.content


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def generate(self, prompt: str) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in r.content if b.type == "text")
