from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

from docx import Document
from pypdf import PdfReader


WHITESPACE_PATTERN = re.compile(r"\s+")
SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
SUPPORTED_EXTENSIONS = {".docx", ".pdf"}


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_name: str
    text: str


def _normalize_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def _slugify(value: str) -> str:
    slug = SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    return slug or "document"


def extract_text_from_docx(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))
    blocks: list[str] = []

    for paragraph in document.paragraphs:
        cleaned = _normalize_text(paragraph.text)
        if cleaned:
            blocks.append(cleaned)

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                cleaned = _normalize_text(cell.text)
                if cleaned:
                    blocks.append(cleaned)

    return "\n".join(blocks)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    blocks: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        cleaned = _normalize_text(page_text)
        if cleaned:
            blocks.append(cleaned)

    return "\n".join(blocks)


def extract_text_from_file(file_name: str, file_bytes: bytes) -> str:
    extension = Path(file_name).suffix.lower()

    if extension == ".docx":
        return extract_text_from_docx(file_bytes)
    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported file type '{extension}'. Supported types: {supported}")


def chunk_text(text: str, chunk_size: int = 220, chunk_overlap: int = 40) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break

    return chunks


def _build_chunks_from_text(
    file_name: str,
    text: str,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> list[DocumentChunk]:
    text_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    source_slug = _slugify(file_name)
    chunks: list[DocumentChunk] = []

    for index, chunk_value in enumerate(text_chunks, start=1):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{source_slug}-chunk-{index}",
                source_name=file_name,
                text=chunk_value,
            )
        )

    return chunks


def build_chunks_from_file(
    file_name: str,
    file_bytes: bytes,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> list[DocumentChunk]:
    text = extract_text_from_file(file_name=file_name, file_bytes=file_bytes)
    return _build_chunks_from_text(
        file_name=file_name,
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_chunks_from_docx(
    file_name: str,
    file_bytes: bytes,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> list[DocumentChunk]:
    text = extract_text_from_docx(file_bytes)
    return _build_chunks_from_text(
        file_name=file_name,
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
