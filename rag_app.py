"""
Multi-PDF RAG App with Sources
Run: streamlit run rag_app.py
Needs: OPENAI_API_KEY in the environment (optional: OPENAI_CHAT_MODEL)
"""
import os
import re
import hashlib
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ---------------- Config ----------------
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE = 800        # ~1 paragraph; keeps one answer to 1-2 pages
CHUNK_OVERLAP = 120
TOP_K = 5
SCATTER_WARN = 4        # warn if one answer cites this many distinct pages

st.set_page_config(page_title="Multi-PDF RAG", page_icon="📚")
st.title("📚 Multi-PDF RAG with Sources")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Export it in your terminal and restart.")
    st.stop()


# ---------------- Indexing ----------------
def fingerprint(files) -> str:
    """Hash of names + bytes, so a new upload set triggers a rebuild."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.name):
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()


def build_store(files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks, empty_files = [], []

    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            path = os.path.join(tmp, f.name)
            with open(path, "wb") as out:
                out.write(f.getvalue())

            pages = PyPDFLoader(path).load()
            for p in pages:
                # Replace temp path with real file name; make page 1-based
                p.metadata["source"] = f.name
                p.metadata["page"] = int(p.metadata.get("page", 0)) + 1

            file_chunks = [
                c for c in splitter.split_documents(pages) if c.page_content.strip()
            ]
            if not file_chunks:
                empty_files.append(f.name)   # likely scanned / image-only PDF
            chunks.extend(file_chunks)

    if not chunks:
        return None, 0, empty_files
    store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return store, len(chunks), empty_files


# ---------------- Answering ----------------
SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources below.

Rules:
1. Cite every factual claim with its source tag, e.g. [S1] or [S2][S4].
2. If sources disagree, do NOT pick one. Start a line with "⚠️ Conflict:" and
   state what each source says, with its tag.
3. If the answer is not in the sources, reply exactly:
   "Not found in the uploaded documents."
4. Be concise."""


def format_context(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(
            f"[S{i}] ({d.metadata['source']}, page {d.metadata['page']})\n"
            f"{d.page_content}"
        )
    return "\n\n".join(blocks)


def answer(question: str):
    retriever = st.session_state.store.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    resp = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Sources:\n{format_context(docs)}\n\nQuestion: {question}"),
    ])
    text = resp.content

    # Keep only sources the model actually cited (fallback: all retrieved)
    cited_ids = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", text)})
    cited = [docs[i - 1] for i in cited_ids if 0 < i <= len(docs)]
    if not cited and "Not found" not in text:
        cited = docs

    # Deduplicate file+page, preserve order
    seen, sources = set(), []
    for d in cited:
        key = (d.metadata["source"], d.metadata["page"])
        if key not in seen:
            seen.add(key)
            sources.append(key)

    return text, sources, docs


def render_answer(text, sources, docs):
    st.markdown(text)
    if sources:
        st.caption("**Sources:** " + " | ".join(f"{s} - page {p}" for s, p in sources))
        if len(sources) >= SCATTER_WARN:
            st.caption(
                f"ℹ️ This answer draws on {len(sources)} different pages — "
                "the evidence is scattered; consider a larger chunk size."
            )
    with st.expander("Retrieved chunks"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**[S{i}] {d.metadata['source']} - page {d.metadata['page']}**")
            st.text(d.page_content[:600])


# ---------------- UI ----------------
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if files:
    fp = fingerprint(files)
    if st.session_state.get("fingerprint") != fp:
        with st.spinner(f"Indexing {len(files)} file(s)..."):
            store, n, empty = build_store(files)
        st.session_state.fingerprint = fp
        st.session_state.store = store
        st.session_state.n_chunks = n
        st.session_state.file_names = [f.name for f in files]
        st.session_state.history = []
        st.session_state.empty_files = empty

if st.session_state.get("store"):
    st.write("**Uploaded:** " + ", ".join(st.session_state.file_names))
    st.success(f"Indexed {st.session_state.n_chunks} chunks.")
    if st.session_state.empty_files:
        st.warning(
            "No text extracted from: " + ", ".join(st.session_state.empty_files)
            + " (scanned PDFs need OCR)."
        )

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            render_answer(turn["a"], turn["sources"], turn["docs"])

    question = st.chat_input("Ask a question across all uploaded PDFs")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                text, sources, docs = answer(question)
            render_answer(text, sources, docs)
        st.session_state.history.append(
            {"q": question, "a": text, "sources": sources, "docs": docs}
        )
elif files:
    st.error("No text could be extracted from any uploaded PDF.")
else:
    st.info("Upload one or more PDFs to begin.")