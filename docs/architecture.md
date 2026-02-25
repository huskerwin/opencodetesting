# Architecture and Runtime Flow

This document explains how the document chatbot works end to end.

## High-Level Architecture

The app uses a retrieval-augmented generation (RAG) pattern:

1. Parse uploaded files into plain text
2. Split text into chunks
3. Build an in-memory TF-IDF index over chunks
4. Retrieve top chunks for each question
5. Ask the LLM to answer using only those chunks

```mermaid
flowchart LR
    U[User] --> UI[Streamlit UI\napp.py]
    UI --> DOCS[Document parser/chunker\nchatbot/documents.py]
    DOCS --> IDX[In-memory TF-IDF index\nchatbot/retrieval.py]
    UI --> SEARCH[search(query)]
    SEARCH --> IDX
    SEARCH --> LLM[Answer generator\nchatbot/llm.py]
    LLM --> OPENAI[OpenAI Chat Completions API\noptional]
    LLM --> FALLBACK[Retrieval-only fallback\nno API key]
    LLM --> UI
    UI --> U
```

## Upload and Indexing Sequence

This runs when the user clicks **Process documents**.

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI (app.py)
    participant Docs as documents.py
    participant Index as retrieval.py

    User->>UI: Upload .docx/.pdf and click Process documents
    UI->>UI: Read chunk_size and chunk_overlap
    loop For each uploaded file
        UI->>Docs: build_chunks_from_file(file_name, file_bytes)
        alt .docx
            Docs->>Docs: extract_text_from_docx()
        else .pdf
            Docs->>Docs: extract_text_from_pdf()
        end
        Docs->>Docs: chunk_text()
        Docs-->>UI: list[DocumentChunk]
    end
    UI->>Index: InMemoryTfidfIndex(all_chunks)
    Index->>Index: _build() -> IDF + vectors + norms
    Index-->>UI: ready index
    UI->>UI: Store chunks/index in session_state
```

## Question Answering Sequence

This runs when the user sends a chat message.

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI (app.py)
    participant Index as InMemoryTfidfIndex
    participant Answer as AnswerGenerator
    participant OpenAI as OpenAI API

    User->>UI: Ask question
    UI->>Index: search(question, top_k=5)
    Index->>Index: tokenize + cosine similarity over TF-IDF vectors
    Index-->>UI: list[SearchResult]
    UI->>Answer: answer(question, results, history)
    alt OPENAI_API_KEY configured
        Answer->>Answer: build_context(results)
        Answer->>OpenAI: chat.completions.create(...)
        OpenAI-->>Answer: grounded response text
    else missing API key or API error
        Answer->>Answer: _fallback_answer(results)
    end
    Answer-->>UI: final answer string
    UI->>UI: Render answer + source chunk previews
    UI-->>User: Chat response
```

## Module Responsibilities

- `app.py`: Streamlit UI, session state, upload workflow, chat loop, and source rendering.
- `chatbot/documents.py`: File parsing (`.docx`, `.pdf`), text normalization, chunking, and `DocumentChunk` creation.
- `chatbot/retrieval.py`: Tokenization, TF-IDF vector build, cosine similarity search, and prompt context assembly.
- `chatbot/llm.py`: LLM orchestration, conversation history inclusion, and retrieval-only fallback mode.

## Core Data Structures

```mermaid
classDiagram
    class DocumentChunk {
      +str chunk_id
      +str source_name
      +str text
    }

    class SearchResult {
      +DocumentChunk chunk
      +float score
    }

    class InMemoryTfidfIndex {
      +chunks: list[DocumentChunk]
      +search(query, top_k, min_score) list[SearchResult]
      -_idf: dict[str, float]
      -_vectors: list[dict[str, float]]
      -_norms: list[float]
    }

    SearchResult --> DocumentChunk
    InMemoryTfidfIndex --> DocumentChunk
    InMemoryTfidfIndex --> SearchResult
```

## Retrieval Details

`InMemoryTfidfIndex` uses standard TF-IDF weighting with cosine similarity:

- `idf(term) = log((N + 1) / (df + 1)) + 1`
- term frequency is normalized by the max term count within each chunk
- similarity is cosine between query vector and chunk vector
- results below `min_score` are filtered (default `0.05`)

## State and Lifecycle

The app stores runtime state in `st.session_state`:

- `chunks`: all parsed `DocumentChunk` objects from current uploads
- `index`: current `InMemoryTfidfIndex`
- `messages`: chat history for rendering and short conversational context
- `answer_generator`: configured `AnswerGenerator` instance

All state is in-memory. Restarting Streamlit clears indexed documents and chat history.

## Configuration

Environment variables loaded from `.env`:

- `OPENAI_API_KEY`: optional; enables full LLM answers
- `OPENAI_MODEL`: optional; defaults to `gpt-4o-mini`

If no API key is set, the chatbot still retrieves relevant chunks and returns excerpt-based fallback responses.

## Known Limitations

- PDF support is text extraction only; scanned/image PDFs need OCR.
- Index is in-memory (not persisted).
- Retrieval is lexical TF-IDF (not semantic embeddings).
- No authentication or per-user document isolation yet.

## Extension Points

- Add OCR for scanned PDFs (for example, `pytesseract` + `pdf2image`).
- Replace TF-IDF with embeddings + vector database.
- Persist per-document indexes to disk or external storage.
- Add citations with exact page/paragraph metadata.
- Add auth and multi-tenant storage for production use.
