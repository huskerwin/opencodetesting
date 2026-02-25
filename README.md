# Document Chatbot

This project is a Streamlit chatbot that lets users upload documents and ask questions about the content.

## What it does

- Upload one or more `.docx` or `.pdf` files
- Extract text and split it into searchable chunks
- Retrieve the most relevant chunks for each question
- Generate an answer using OpenAI (if configured)
- Show source chunks used to answer each question

If no OpenAI key is configured, the app still works in retrieval mode and returns the closest excerpts.

## Tech stack

- Python 3.11+
- Streamlit for the chat UI
- `python-docx` for reading Word files
- `pypdf` for reading PDF files
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

3. (Optional but recommended) configure OpenAI:

```bash
copy .env.example .env
```

Then update `.env` with your API key.

4. Start the app:

```bash
streamlit run app.py
```

5. In the UI:

- Upload `.docx` or `.pdf` file(s)
- Click **Process documents**
- Ask questions in the chat box

## Notes

- Current upload support is `.docx` and text-based `.pdf` files.
- Scanned/image-only PDFs require OCR, which is not included in this MVP.
- Large documents are chunked by words for better retrieval.
- Retrieval is in-memory; restarting the app clears indexed documents.
