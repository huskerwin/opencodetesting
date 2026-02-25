# Document Chatbot

[![CI](https://github.com/huskerwin/opencodetesting/actions/workflows/ci.yml/badge.svg)](https://github.com/huskerwin/opencodetesting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is a Streamlit chatbot that lets users upload documents and ask questions about the content.

## Documentation

- Full architecture and runtime diagrams: `docs/architecture.md`
- Usage guide: `docs/usage.md`
- Deployment notes: `docs/deployment.md`
- Changelog: `CHANGELOG.md`

## What it does

- Upload one or more `.docx` or `.pdf` files
- Extract text and split it into searchable chunks
- Run OCR on scanned PDF pages when needed
- Retrieve the most relevant chunks for each question
- Generate an answer using OpenAI (if configured)
- Show source chunks used to answer each question

If no OpenAI key is configured, the app still works in retrieval mode and returns the closest excerpts.

## Tech stack

- Python 3.11+
- Streamlit for the chat UI
- `python-docx` for reading Word files
- `pypdf` for reading PDF files
- `pypdfium2` + `pytesseract` for OCR on scanned PDF pages
- In-memory TF-IDF retrieval (no external vector DB)
- OpenAI Chat Completions API (optional, for best answers)

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (required for scanned/image PDFs):

- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
- Then either add Tesseract to PATH or set `TESSERACT_CMD` in `.env`
- The app auto-detects common Windows install paths if available

4. (Optional but recommended) configure environment variables:

```bash
copy .env.example .env
```

Then update `.env` with your API key and OCR settings if needed.

5. Start the app:

```bash
streamlit run app.py
```

6. In the UI:

- Upload `.docx` or `.pdf` file(s)
- Click **Process documents**
- Ask questions in the chat box

## Running tests

Run the automated unit test suite with:

```bash
python -m pytest -q
```

The tests cover core ingestion, OCR fallback logic, retrieval ranking, LLM
fallback behavior, and upload processing workflows.

## Project standards

- License: `LICENSE` (MIT)
- Contributing: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Support channels: `SUPPORT.md`

## Notes

- Current upload support is `.docx` and `.pdf` files.
- Scanned/image-only PDFs are supported through OCR (Tesseract required).
- OCR runs automatically for PDF pages with little or no extractable text.
- Large documents are chunked by words for better retrieval.
- Retrieval is in-memory; restarting the app clears indexed documents.
