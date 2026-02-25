from __future__ import annotations

from chatbot.documents import DocumentChunk
from chatbot.retrieval import InMemoryTfidfIndex, SearchResult, build_context, tokenize


def _chunk(chunk_id: str, text: str, source: str = "sample.docx") -> DocumentChunk:
    return DocumentChunk(chunk_id=chunk_id, source_name=source, text=text)


def test_tokenize_normalizes_terms() -> None:
    terms = tokenize("Hello, world! It's high-quality text.")
    assert terms == ["hello", "world", "it's", "high-quality", "text"]


def test_search_returns_empty_for_empty_index() -> None:
    index = InMemoryTfidfIndex([])
    assert index.search("anything") == []


def test_search_returns_empty_for_unknown_query_terms() -> None:
    index = InMemoryTfidfIndex([_chunk("c1", "apple banana")])
    assert index.search("kiwi mango") == []


def test_search_ranks_most_relevant_chunk_first() -> None:
    chunks = [
        _chunk("c1", "apple banana fruit"),
        _chunk("c2", "car engine vehicle"),
        _chunk("c3", "apple pie apple dessert"),
    ]
    index = InMemoryTfidfIndex(chunks)

    results = index.search("apple pie", top_k=3, min_score=0.0)

    assert results[0].chunk.chunk_id == "c3"
    assert results[0].score >= results[1].score


def test_search_respects_top_k_limit() -> None:
    chunks = [_chunk(f"c{i}", f"topic {i} shared term") for i in range(6)]
    index = InMemoryTfidfIndex(chunks)

    results = index.search("shared term", top_k=2, min_score=0.0)

    assert len(results) == 2


def test_search_respects_min_score_threshold() -> None:
    index = InMemoryTfidfIndex([_chunk("c1", "alpha beta gamma")])
    assert index.search("alpha", min_score=1.1) == []


def test_build_context_formats_headers_and_truncates_by_length() -> None:
    first = SearchResult(chunk=_chunk("chunk-1", "A" * 120), score=0.9)
    second = SearchResult(chunk=_chunk("chunk-2", "B" * 40), score=0.8)

    context = build_context([first, second], max_chars=60)

    assert "chunk-1" in context
    assert "chunk-2" not in context
    assert "score=0.900" in context


def test_dot_product_works_for_different_vector_sizes() -> None:
    value = InMemoryTfidfIndex._dot(
        {"alpha": 1.0, "beta": 2.0, "gamma": 4.0},
        {"beta": 3.0, "gamma": 5.0},
    )
    assert value == (2.0 * 3.0) + (4.0 * 5.0)
