"""In-memory lexical retrieval based on TF-IDF and cosine similarity.

The app is intentionally lightweight and does not require a vector database.
This module provides deterministic retrieval over document chunks so answers can
be grounded in user-uploaded content.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from chatbot.documents import DocumentChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


def tokenize(text: str) -> list[str]:
    """Tokenize text into normalized terms for indexing and querying."""

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


@dataclass(frozen=True)
class SearchResult:
    """Single retrieval hit with its corresponding relevance score."""

    chunk: DocumentChunk
    score: float


class InMemoryTfidfIndex:
    """Simple TF-IDF index over `DocumentChunk` records."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        """Build an index from pre-chunked document text."""

        self.chunks = chunks
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._norms: list[float] = []
        self._build()

    def _build(self) -> None:
        """Precompute IDF values and chunk vectors for fast search."""

        if not self.chunks:
            return

        tokenized_chunks = [tokenize(chunk.text) for chunk in self.chunks]
        doc_frequencies: Counter[str] = Counter()

        for terms in tokenized_chunks:
            doc_frequencies.update(set(terms))

        num_docs = len(self.chunks)
        self._idf = {
            term: math.log((num_docs + 1) / (doc_freq + 1)) + 1.0
            for term, doc_freq in doc_frequencies.items()
        }

        for terms in tokenized_chunks:
            vector = self._vectorize_terms(terms)
            norm = math.sqrt(sum(weight * weight for weight in vector.values()))
            self._vectors.append(vector)
            self._norms.append(norm)

    def _vectorize_terms(self, terms: list[str]) -> dict[str, float]:
        """Convert token sequences into TF-IDF vectors."""

        term_counts: Counter[str] = Counter(terms)
        if not term_counts:
            return {}

        max_frequency = max(term_counts.values())
        vector: dict[str, float] = {}

        for term, count in term_counts.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = count / max_frequency
            vector[term] = tf * idf

        return vector

    @staticmethod
    def _dot(left: dict[str, float], right: dict[str, float]) -> float:
        """Compute dot product while iterating over the smaller dictionary."""

        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(term, 0.0) for term, weight in left.items())

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> list[SearchResult]:
        """Return the top matching chunks for a natural language query."""

        if not self.chunks:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        query_vector = self._vectorize_terms(query_terms)
        if not query_vector:
            return []

        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        if query_norm == 0.0:
            return []

        scored: list[SearchResult] = []

        for index, chunk in enumerate(self.chunks):
            chunk_norm = self._norms[index]
            if chunk_norm == 0.0:
                continue

            score = self._dot(query_vector, self._vectors[index]) / (query_norm * chunk_norm)
            if score >= min_score:
                scored.append(SearchResult(chunk=chunk, score=score))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


def build_context(results: list[SearchResult], max_chars: int = 8000) -> str:
    """Format retrieval hits into a context string for LLM prompting.

    Each block includes chunk metadata so the assistant can cite sources.
    """

    sections: list[str] = []
    total_chars = 0

    for result in results:
        header = (
            f"[{result.chunk.chunk_id} | {result.chunk.source_name} | "
            f"score={result.score:.3f}]"
        )
        block = f"{header}\n{result.chunk.text}".strip()
        estimated_length = len(block) + 2

        if sections and total_chars + estimated_length > max_chars:
            break

        sections.append(block)
        total_chars += estimated_length

    return "\n\n".join(sections)
