# Usage Guide

This guide describes how to run and operate the Document Chatbot in local
development or internal environments.

## Prerequisites

- Python 3.11+
- Tesseract OCR installed (for scanned PDFs)
- OpenAI API key (optional but recommended)

## Setup

1. Create a virtual environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
copy .env.example .env
```

Then edit `.env`.

## Environment variables

- `OPENAI_API_KEY`: enables LLM-based answers
- `OPENAI_MODEL`: chat model name, defaults to `gpt-4o-mini`
- `TESSERACT_CMD`: optional explicit path to `tesseract` binary
- `OCR_LANGUAGE`: OCR language code (default `eng`)

## Running the app

```bash
streamlit run app.py
```

## Typical workflow

1. Upload one or more `.docx` and/or `.pdf` files
2. Choose chunk size and chunk overlap
3. Keep **Enable OCR for scanned PDFs** enabled if scanned docs are expected
4. Click **Process documents**
5. Ask a question in the chat box
6. Expand **Sources** to inspect supporting chunks

## Behavior notes

- OCR runs only on PDF pages with little/no directly extractable text
- If OpenAI is not configured, the app returns retrieval-based excerpts
- All indexing state is in memory and resets when the app restarts

## Troubleshooting

### OCR errors

- Ensure Tesseract is installed
- Add Tesseract to PATH, or set `TESSERACT_CMD` in `.env`
- Verify OCR language packs are installed for your selected `OCR_LANGUAGE`

### No answers found

- Reprocess documents after changing chunk settings
- Ask more specific questions with terms present in the source documents
- Confirm files were parsed successfully in sidebar feedback
