"""Shared RAG pipeline.

Imported by both `app.py` and the eval harness, on purpose: if the harness
built its own retrieval path, it would be scoring a system that isn't the one
users talk to. One code path, measured and served.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_GLOB = str(REPO_ROOT / "data" / "Everstorm_*.pdf")
INDEX_CACHE = REPO_ROOT / ".eval_cache"

# Fields that change the vectors on disk. Anything else (k, prompt, llm) is
# applied at query time and reuses the same index.
INDEX_FIELDS = ("chunk_size", "chunk_overlap", "embedding_model")


# --------------------------------------------------------------------------
# text normalization (shared by context formatting and answer-key matching)
# --------------------------------------------------------------------------

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}


def normalize(text: str) -> str:
    """Lowercase, unify dashes/quotes, strip bullets, collapse whitespace.

    The PDFs use en-dashes, curly quotes and bullet glyphs that a naive
    substring match would miss, so every comparison goes through here.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DASHES).translate(_QUOTES)
    text = text.replace("​", "").replace(" ", " ")
    text = re.sub(r"[•●▪◦­]", " ", text)
    text = text.lower()
    text = re.sub(r"\s*-\s*", "-", text)  # "5 - 7" and "5-7" compare equal
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def source_label(doc) -> str:
    """Human-readable citation label, e.g. 'Everstorm_Shipping....pdf p.2'."""
    src = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")
    return f"{src} p.{page + 1}" if isinstance(page, int) else src


def format_context(docs) -> str:
    """Render retrieved chunks with their source label attached.

    The label has to be inside the context string: the model can only cite
    metadata it can actually see, and previously it saw page_content alone.
    """
    return "\n\n".join(
        f"[source: {source_label(d)}]\n{d.page_content}" for d in docs
    )


# --------------------------------------------------------------------------
# index build / cache
# --------------------------------------------------------------------------

def index_fingerprint(config: dict) -> str:
    payload = {f: config[f] for f in INDEX_FIELDS}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def load_chunks(config: dict):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        raise FileNotFoundError(f"no source PDFs matched {DATA_GLOB}")

    raw_docs = []
    for path in paths:
        raw_docs.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    return splitter.split_documents(raw_docs)


def get_embeddings(config: dict):
    """Embedding model for this config.

    Isolated in one function so offline tests can swap in a stub without
    touching the rest of the pipeline.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config["embedding_model"])


def build_or_load_index(config: dict, rebuild: bool = False):
    """Return a FAISS store for this config, building and caching on miss."""
    from langchain_community.vectorstores import FAISS

    fingerprint = index_fingerprint(config)
    cache_dir = INDEX_CACHE / f"idx_{fingerprint}"
    embeddings = get_embeddings(config)

    if cache_dir.exists() and not rebuild:
        return FAISS.load_local(
            str(cache_dir), embeddings, allow_dangerous_deserialization=True
        )

    chunks = load_chunks(config)
    store = FAISS.from_documents(chunks, embeddings)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(cache_dir))
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {**{f: config[f] for f in INDEX_FIELDS}, "n_chunks": len(chunks)}, indent=2
        )
    )
    return store


def get_retriever(config: dict, rebuild: bool = False):
    store = build_or_load_index(config, rebuild=rebuild)
    if config.get("search_type") == "mmr":
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config["k"], "lambda_mult": config.get("mmr_lambda", 0.5)},
        )
    return store.as_retriever(search_kwargs={"k": config["k"]})


def get_llm(config: dict):
    from langchain_ollama import OllamaLLM

    return OllamaLLM(model=config["llm_model"], temperature=config["temperature"])


def answer(question: str, config: dict, retriever=None, llm=None, prompt=None):
    """One full RAG pass. Returns the answer plus the docs that produced it."""
    from langchain_core.prompts import ChatPromptTemplate

    retriever = retriever if retriever is not None else get_retriever(config)
    llm = llm if llm is not None else get_llm(config)
    prompt = prompt if prompt is not None else ChatPromptTemplate.from_template(
        config["system_template"]
    )

    docs = retriever.invoke(question)
    context = format_context(docs)
    messages = prompt.format_messages(context=context, question=question)
    return {"answer": llm.invoke(messages), "docs": docs, "context": context}
