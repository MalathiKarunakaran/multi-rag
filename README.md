# Multi-PDF RAG App with Sources

A LangChain + Streamlit app that answers questions across several PDFs at once.
All uploads are chunked, embedded and stored in a single FAISS vector database.
Every answer comes back with its **source: file name and page number**, so the reader can verify it.

## Features

- Upload multiple PDFs in one go
- Chunks and embeds all files into **one FAISS store** (kept in `st.session_state`)
- Reports the number of chunks indexed
- Retrieval across every uploaded document
- Every answer shows `Sources: file.pdf - page N`, taken from chunk metadata
- **Conflict detection**: if documents disagree, the answer says so explicitly instead of silently picking one
- Returns `Not found in the uploaded documents.` when the answer isn't in the PDFs
- "Retrieved chunks" panel to inspect the exact text and metadata behind each answer
- Warns about scanned/image-only PDFs that yield no text

## Tech Stack

Python · LangChain · Streamlit · FAISS · OpenAI (embeddings + chat) · pypdf

## Setup

```bash
git clone https://github.com/MalathiKarunakaran/multi-rag.git
cd multi-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder (see `.env.example`):

```
OPENAI_API_KEY=your-key-here
OPENAI_CHAT_MODEL=gpt-4o-mini
```

`.env` is git-ignored — never commit your real key.

## Run

```bash
streamlit run rag_app.py
```

Open the URL shown in the terminal (usually http://localhost:8501), upload PDFs and ask questions.

## Example

```
Uploaded: policy.pdf, handbook.pdf, faq.pdf
Indexed 412 chunks.

Q: How many leave days do new joiners get?
A: New joiners get 12 days in their first year. [S1][S3]
   Sources: handbook.pdf - page 7 | policy.pdf - page 2
```

## Design Decisions

| Decision | Why |
|---|---|
| Chunk size 800, overlap 120 | About one paragraph, so an answer usually cites 1–2 pages. If one answer cites 4+ distinct pages, the app flags it — a sign the chunks are too small. |
| Top-k = 5 | Enough context for cross-document answers without diluting it. |
| Numbered source tags `[S1]…[S5]` | The model must cite each claim; only the sources actually cited are listed under the answer. |
| Explicit conflict rule in the prompt | With several documents in one store, the honest answer is sometimes "these two disagree" — the app surfaces that. |
| Page number +1 | PyPDFLoader pages are 0-indexed; shown 1-based to match the PDF viewer. |
| Real file name in metadata | Uploads are saved to a temp folder; the temp path is replaced with the original file name. |
| Re-index on upload change | A hash of file names and contents triggers a rebuild when the upload set changes. |
| Temperature 0 | Deterministic, grounded answers. |

## Limitations

- RAG answers specific questions well; "list everything in the document" questions return only what the top-k chunks contain.
- Scanned PDFs need OCR before upload.
- The index lives in memory and is rebuilt each session.

## Project Structure

```
multi-rag/
├── rag_app.py         # the complete app
├── requirements.txt
├── .env.example       # template for your API key
├── .gitignore
└── README.md
```
