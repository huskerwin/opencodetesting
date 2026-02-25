from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv

from chatbot.retrieval import SearchResult, build_context


try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None


SYSTEM_PROMPT = """
You are a careful assistant for question-answering over uploaded documents.
Only use the provided context snippets.
If the context does not contain the answer, say you do not know.
When possible, include short citations in parentheses using chunk ids.
""".strip()


class AnswerGenerator:
    def __init__(self) -> None:
        load_dotenv()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._client = OpenAI(api_key=self.api_key) if self.api_key and OpenAI else None

    def answer(
        self,
        question: str,
        results: Sequence[SearchResult],
        history: Sequence[dict[str, str]],
    ) -> str:
        if not results:
            return "I could not find relevant text in the uploaded documents for that question."

        if self._client is None:
            return self._fallback_answer(results)

        context = build_context(list(results))

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for message in history[-8:]:
            role = message.get("role")
            content = message.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        user_prompt = (
            "Answer the question using only the context snippets below.\n"
            "If information is missing, say you do not know.\n\n"
            f"Question:\n{question}\n\n"
            f"Context snippets:\n{context}"
        )
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            )
            answer = response.choices[0].message.content
            if answer:
                return answer.strip()
        except Exception:
            return self._fallback_answer(results)

        return self._fallback_answer(results)

    def _fallback_answer(self, results: Sequence[SearchResult]) -> str:
        lines = [
            "I found relevant passages, but no LLM is configured.",
            "Set OPENAI_API_KEY in your .env file for full conversational answers.",
            "",
            "Closest excerpts:",
        ]

        for result in results[:3]:
            excerpt = result.chunk.text.replace("\n", " ").strip()
            if len(excerpt) > 320:
                excerpt = f"{excerpt[:317]}..."
            lines.append(
                f"- {result.chunk.source_name} ({result.chunk.chunk_id}, "
                f"score={result.score:.3f}): {excerpt}"
            )

        return "\n".join(lines)
