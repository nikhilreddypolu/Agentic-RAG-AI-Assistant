import os
from typing import List
from pathlib import Path

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# global index and docs cache
_index = None
_docs: List[str] = []
_model = None


def _index_paths():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "index.faiss", data_dir / "docs.json"


def _save_index():
    """Persist FAISS index and docs to disk."""
    global _index, _docs
    index_path, docs_path = _index_paths()
    if _index is not None:
        faiss.write_index(_index, str(index_path))
    if _docs is not None:
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(_docs, f, ensure_ascii=False)


def _load_index_if_exists():
    """Load persisted FAISS index and docs if they exist."""
    global _index, _docs, _model
    index_path, docs_path = _index_paths()
    if index_path.exists() and docs_path.exists():
        # ensure model is available for queries
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _index = faiss.read_index(str(index_path))
        with open(docs_path, "r", encoding="utf-8") as f:
            _docs = json.load(f)
        return True
    return False


def _build_index():
    """Build FAISS index from PDFs and persist it to disk if not present."""
    global _index, _docs, _model
    if _index is not None:
        return

    # try load existing
    if _load_index_if_exists():
        return

    # load model
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    # walk through data directory and read PDFs
    data_dir = Path(__file__).parent / "data"
    texts: List[str] = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            path = data_dir / fname
            doc = fitz.open(path)
            for page in doc:
                texts.append(page.get_text())

    # simple chunking
    chunks: List[str] = []
    chunk_size = 500
    overlap = 50
    for text in texts:
        tokens = text.split()
        i = 0
        while i < len(tokens):
            chunk = " ".join(tokens[i : i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
    _docs = chunks

    if len(chunks) == 0:
        # nothing to index
        _index = None
        return

    # embed
    embeddings = _model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(np.array(embeddings, dtype=np.float32))

    # persist
    try:
        _save_index()
    except Exception:
        # non-fatal if persistence fails
        pass


def get_relevant_docs(query: str, top_k: int = 3) -> List[str]:
    """Return top_k most relevant document chunks for the query."""
    if not query:
        return []
    _build_index()
    query_vec = _model.encode([query], convert_to_numpy=True)
    distances, indices = _index.search(query_vec, top_k)
    results: List[str] = []
    for idx in indices[0]:
        if idx < len(_docs):
            results.append(_docs[idx])
    return results
