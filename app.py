from __future__ import annotations

import streamlit as st

from chatbot.documents import build_chunks_from_file
from chatbot.llm import AnswerGenerator
from chatbot.retrieval import InMemoryTfidfIndex


def _initialize_state() -> None:
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "answer_generator" not in st.session_state:
        st.session_state.answer_generator = AnswerGenerator()


def _show_sources(sources: list[dict[str, str | float]]) -> None:
    if not sources:
        return

    with st.expander("Sources", expanded=False):
        for source in sources:
            st.write(
                f"`{source['chunk_id']}` from `{source['document']}` "
                f"(score {source['score']:.3f})"
            )
            st.caption(source["preview"])


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _show_sources(message.get("sources", []))


def _process_uploads(chunk_size: int, chunk_overlap: int) -> None:
    uploaded_files = st.session_state.get("uploaded_files", [])
    if not uploaded_files:
        st.sidebar.warning("Upload at least one .docx or .pdf file first.")
        return

    all_chunks = []
    failed_files: list[str] = []

    with st.sidebar.spinner("Reading and indexing documents..."):
        for uploaded_file in uploaded_files:
            try:
                chunks = build_chunks_from_file(
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if chunks:
                    all_chunks.extend(chunks)
                else:
                    failed_files.append(uploaded_file.name)
            except Exception:
                failed_files.append(uploaded_file.name)

    if not all_chunks:
        st.sidebar.error("No readable text found in the selected files.")
        return

    st.session_state.chunks = all_chunks
    st.session_state.index = InMemoryTfidfIndex(all_chunks)
    st.session_state.messages = []

    processed_count = len(uploaded_files) - len(failed_files)
    st.sidebar.success(
        f"Indexed {len(all_chunks)} chunk(s) from {processed_count} file(s)."
    )

    if failed_files:
        failed_label = ", ".join(failed_files)
        st.sidebar.warning(f"Could not parse: {failed_label}")


def main() -> None:
    st.set_page_config(page_title="Document Chatbot", page_icon=":books:", layout="wide")
    st.title("Document Chatbot")
    st.caption("Upload .docx or .pdf files and ask questions about their content.")

    _initialize_state()

    st.sidebar.header("Document Setup")
    st.sidebar.file_uploader(
        "Upload documents",
        type=["docx", "pdf"],
        accept_multiple_files=True,
        key="uploaded_files",
    )
    chunk_size = st.sidebar.slider(
        "Chunk size (words)",
        min_value=120,
        max_value=500,
        value=220,
        step=20,
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk overlap (words)",
        min_value=20,
        max_value=120,
        value=40,
        step=10,
    )

    st.sidebar.button(
        "Process documents",
        use_container_width=True,
        on_click=_process_uploads,
        args=(chunk_size, chunk_overlap),
    )

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.sidebar.success("Chat history cleared.")

    if st.session_state.index is None:
        st.info("Upload and process at least one .docx or .pdf file to start chatting.")

    _render_history()

    question = st.chat_input("Ask a question about your uploaded documents...")
    if not question:
        return

    if st.session_state.index is None:
        st.warning("Please upload and process documents first.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and drafting an answer..."):
            search_results = st.session_state.index.search(question, top_k=5)
            answer = st.session_state.answer_generator.answer(
                question=question,
                results=search_results,
                history=st.session_state.messages[:-1],
            )

        st.markdown(answer)

        sources = []
        for result in search_results:
            preview = result.chunk.text.replace("\n", " ").strip()
            if len(preview) > 180:
                preview = f"{preview[:177]}..."
            sources.append(
                {
                    "chunk_id": result.chunk.chunk_id,
                    "document": result.chunk.source_name,
                    "score": result.score,
                    "preview": preview,
                }
            )

        _show_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    main()
