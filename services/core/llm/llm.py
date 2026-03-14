# services/core/llm/llm.py
"""
LLM adapter with OpenAI chat completions as default provider.
"""

import time
from typing import Iterable
from openai import OpenAI
from services.core.config import OPENAI_API_KEY, CHAT_MODEL

_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=20.0,
    max_retries=2,
)

SYSTEM_PROMPT = (
    "You are an enterprise assistant. Answer using only the provided context. "
    "If the answer is not in the context, say you do not know. Be concise."
)


def generate_answer(
    question: str,
    context_chunks: Iterable[str],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    context_text = "\n\n".join(context_chunks).strip()
    if not context_text:
        return "I do not know based on the provided context."

    last_exception = None
    for attempt in range(3):
        try:
            response = _client.chat.completions.create(
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
        except Exception as exception:
            last_exception = exception
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))

    raise RuntimeError(f"Answer generation failed: {last_exception}")
