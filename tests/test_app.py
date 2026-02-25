from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import app as app_module
from chatbot.documents import DocumentChunk


class _Spinner:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False


class _SidebarRecorder:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.spinner_messages: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def spinner(self, message: str) -> _Spinner:
        self.spinner_messages.append(message)
        return _Spinner()


class _SessionState(dict[str, object]):
    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state = _SessionState()
        self.sidebar = _SidebarRecorder()


@dataclass
class _UploadedFile:
    name: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload


def test_initialize_state_sets_expected_defaults(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake_st = _FakeStreamlit()

    class _FakeAnswerGenerator:
        pass

    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "AnswerGenerator", _FakeAnswerGenerator)

    app_module._initialize_state()

    assert fake_st.session_state["chunks"] == []
    assert fake_st.session_state["index"] is None
    assert fake_st.session_state["messages"] == []
    assert isinstance(fake_st.session_state["answer_generator"], _FakeAnswerGenerator)


def test_process_uploads_warns_when_no_files(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake_st = _FakeStreamlit()
    fake_st.session_state["uploaded_files"] = []
    monkeypatch.setattr(app_module, "st", fake_st)

    app_module._process_uploads(chunk_size=220, chunk_overlap=40, enable_pdf_ocr=True)

    assert fake_st.sidebar.warnings == ["Upload at least one .docx or .pdf file first."]


def test_process_uploads_builds_index_and_clears_messages(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake_st = _FakeStreamlit()
    fake_st.session_state["uploaded_files"] = [_UploadedFile("contract.pdf", b"pdf-bytes")]
    fake_st.session_state["messages"] = [{"role": "user", "content": "old"}]
    monkeypatch.setattr(app_module, "st", fake_st)

    captured: dict[str, object] = {}

    def fake_build_chunks_from_file(
        file_name: str,
        file_bytes: bytes,
        chunk_size: int,
        chunk_overlap: int,
        enable_pdf_ocr: bool,
    ) -> list[DocumentChunk]:
        captured["file_name"] = file_name
        captured["file_bytes"] = file_bytes
        captured["chunk_size"] = chunk_size
        captured["chunk_overlap"] = chunk_overlap
        captured["enable_pdf_ocr"] = enable_pdf_ocr
        return [
            DocumentChunk(
                chunk_id="contract-pdf-chunk-1",
                source_name="contract.pdf",
                text="important clause",
            )
        ]

    class _FakeIndex:
        def __init__(self, chunks: list[DocumentChunk]) -> None:
            self.chunks = chunks

    monkeypatch.setattr(app_module, "build_chunks_from_file", fake_build_chunks_from_file)
    monkeypatch.setattr(app_module, "InMemoryTfidfIndex", _FakeIndex)

    app_module._process_uploads(chunk_size=180, chunk_overlap=30, enable_pdf_ocr=False)

    assert captured["enable_pdf_ocr"] is False
    assert captured["chunk_size"] == 180
    assert captured["chunk_overlap"] == 30
    assert fake_st.session_state["messages"] == []
    chunk_list = cast(list[DocumentChunk], fake_st.session_state["chunks"])
    assert chunk_list[0].chunk_id == "contract-pdf-chunk-1"
    assert isinstance(fake_st.session_state["index"], _FakeIndex)
    assert fake_st.sidebar.successes == ["Indexed 1 chunk(s) from 1 file(s)."]


def test_process_uploads_reports_errors_when_every_file_fails(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake_st = _FakeStreamlit()
    fake_st.session_state["uploaded_files"] = [_UploadedFile("scan.pdf", b"pdf-bytes")]
    monkeypatch.setattr(app_module, "st", fake_st)

    def raise_parse_error(**_kwargs: object) -> list[DocumentChunk]:
        raise RuntimeError("OCR unavailable")

    monkeypatch.setattr(app_module, "build_chunks_from_file", raise_parse_error)

    app_module._process_uploads(chunk_size=220, chunk_overlap=40, enable_pdf_ocr=True)

    assert fake_st.sidebar.errors == ["No readable text found in the selected files."]
