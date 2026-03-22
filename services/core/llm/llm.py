# services/core/llm/llm.py
"""
LLM adapter supporting OpenAI or Ollama.
"""

import time
from typing import Iterable

import requests
from openai import OpenAI

from services.core.config import (
    CHAT_MODEL,
    CHAT_PROVIDER,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
)

SYSTEM_PROMPT = (
    "You are an enterprise assistant. Answer using only the provided context. "
    "If the answer is not in the context, say you do not know. Be concise."
)


def _openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=20.0,
        max_retries=2,
    )


def _generate_openai_answer(
    question: str,
    context_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
    client = _openai_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context_text}",
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _generate_ollama_answer(
    question: str,
    context_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context_text}",
            },
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    message = data.get("message", {})
    return (message.get("content") or "").strip()


def generate_answer(
    question: str,
    context_chunks: Iterable[str],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    context_text = "\n\n".join(context_chunks).strip()
    if not context_text:
        return "I do not know based on the provided context."

    last_exception = None
    for attempt in range(3):
        try:
            if CHAT_PROVIDER == "openai":
                return _generate_openai_answer(
                    question, context_text, max_tokens, temperature
                )
            return _generate_ollama_answer(
                question, context_text, max_tokens, temperature
            )
        except Exception as exception:
            last_exception = exception
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))

    raise RuntimeError(f"Answer generation failed: {last_exception}")
