from __future__ import annotations

from types import SimpleNamespace

import pytest

from chatbot.documents import DocumentChunk
from chatbot.llm import AnswerGenerator
from chatbot import llm as llm_module
from chatbot.retrieval import SearchResult


def _result(index: int, text: str, score: float = 0.7) -> SearchResult:
    chunk = DocumentChunk(
        chunk_id=f"doc-chunk-{index}",
        source_name="doc.pdf",
        text=text,
    )
    return SearchResult(chunk=chunk, score=score)


class _FakeCompletions:
    def __init__(self, content: str | None = "Model answer", should_raise: bool = False) -> None:
        self.content = content
        self.should_raise = should_raise
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.should_raise:
            raise RuntimeError("upstream failure")

        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAIFactory:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions
        self.api_keys: list[str] = []

    def __call__(self, api_key: str) -> SimpleNamespace:
        self.api_keys.append(api_key)
        return SimpleNamespace(chat=SimpleNamespace(completions=self.completions))


def test_answer_returns_no_match_message_when_results_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "OpenAI", None)

    generator = AnswerGenerator()
    response = generator.answer("What is this?", results=[], history=[])

    assert response == "I could not find relevant text in the uploaded documents for that question."


def test_answer_falls_back_when_no_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "OpenAI", None)

    generator = AnswerGenerator()
    response = generator.answer(
        "question",
        results=[_result(1, "first excerpt"), _result(2, "second excerpt")],
        history=[],
    )

    assert "Closest excerpts:" in response
    assert "doc-chunk-1" in response


def test_fallback_output_limits_to_three_excerpts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "OpenAI", None)

    generator = AnswerGenerator()
    response = generator.answer(
        "question",
        results=[
            _result(1, "excerpt one"),
            _result(2, "excerpt two"),
            _result(3, "excerpt three"),
            _result(4, "excerpt four"),
        ],
        history=[],
    )

    assert "doc-chunk-1" in response
    assert "doc-chunk-2" in response
    assert "doc-chunk-3" in response
    assert "doc-chunk-4" not in response


def test_answer_uses_openai_and_trims_history(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = _FakeCompletions(content="  Synthesized answer  ")
    factory = _FakeOpenAIFactory(completions)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(llm_module, "OpenAI", factory)

    generator = AnswerGenerator()
    history = [{"role": "user", "content": f"message {index}"} for index in range(10)]

    response = generator.answer(
        "What does the contract say?",
        results=[_result(1, "Contract clause text")],
        history=history,
    )

    assert response == "Synthesized answer"
    assert factory.api_keys == ["test-key"]

    call = completions.calls[0]
    assert call["model"] == "test-model"
    messages = call["messages"]

    history_messages = messages[1:-1]
    assert [entry["content"] for entry in history_messages] == [
        "message 2",
        "message 3",
        "message 4",
        "message 5",
        "message 6",
        "message 7",
        "message 8",
        "message 9",
    ]
    assert "Question:\nWhat does the contract say?" in messages[-1]["content"]


def test_answer_filters_invalid_or_blank_history_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = _FakeCompletions(content="answer")
    factory = _FakeOpenAIFactory(completions)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "OpenAI", factory)

    generator = AnswerGenerator()
    history = [
        {"role": "user", "content": "valid user"},
        {"role": "assistant", "content": "valid assistant"},
        {"role": "tool", "content": "should be ignored"},
        {"role": "user", "content": "   "},
    ]

    generator.answer(
        "question",
        results=[_result(1, "snippet")],
        history=history,
    )

    messages = completions.calls[0]["messages"]
    history_messages = messages[1:-1]
    assert history_messages == [
        {"role": "user", "content": "valid user"},
        {"role": "assistant", "content": "valid assistant"},
    ]


def test_answer_falls_back_on_openai_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = _FakeCompletions(should_raise=True)
    factory = _FakeOpenAIFactory(completions)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "OpenAI", factory)

    generator = AnswerGenerator()
    response = generator.answer(
        "question",
        results=[_result(1, "snippet")],
        history=[],
    )

    assert "Closest excerpts:" in response


def test_answer_falls_back_on_empty_model_content(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = _FakeCompletions(content=None)
    factory = _FakeOpenAIFactory(completions)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "OpenAI", factory)

    generator = AnswerGenerator()
    response = generator.answer(
        "question",
        results=[_result(1, "snippet")],
        history=[],
    )

    assert "Closest excerpts:" in response
