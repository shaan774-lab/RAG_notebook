from dotenv import load_dotenv
load_dotenv()

import logging
from pathlib import Path
import pickle

import faiss
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

BASE_DIR = Path(__file__).parent
VECTOR_DIR = BASE_DIR / "vector_store"
EMBED_FILE = VECTOR_DIR / "embeddings.faiss"
META_FILE = VECTOR_DIR / "metadata.pkl"

# Lazy-loaded globals
_INDEX = None
_CHUNKS = None
_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is None:
        # import heavy library at runtime only
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

def _load_index_and_chunks():
    global _INDEX, _CHUNKS
    if _INDEX is not None and _CHUNKS is not None:
        return _INDEX, _CHUNKS

    if not EMBED_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Vector DB not found. Run backend/ingest.py first.")

    _INDEX = faiss.read_index(str(EMBED_FILE))
    with open(META_FILE, "rb") as f:
        _CHUNKS = pickle.load(f)
    logger.info("Loaded vector DB with %d chunks", len(_CHUNKS))
    return _INDEX, _CHUNKS

def ingest_pdf(file_path):
    """Ingest a single PDF into vector_store (overwrites existing DB)."""
    from PyPDF2 import PdfReader

    txts = []
    reader = PdfReader(str(file_path))
    for p in reader.pages:
        t = p.extract_text()
        if t:
            txts.append(t)

    text = "\n\n".join(txts).strip()
    if not text:
        raise ValueError("No text extracted from PDF")

    # Prefer LangChain splitter but fallback to simple splitter if missing
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
    except Exception:
        logger.warning("langchain splitter unavailable — using local fallback in rag_engine")
        def _simple_split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
            chunks = []
            i = 0
            n = len(text)
            if n == 0:
                return chunks
            step = max(1, chunk_size - chunk_overlap)
            while i < n:
                end = min(i + chunk_size, n)
                chunk = text[i:end].strip()
                if chunk:
                    chunks.append(chunk)
                i += step
            return chunks
        chunks = _simple_split_text(text, chunk_size=500, chunk_overlap=100)

    model = _load_model()
    embeddings = model.encode(chunks, convert_to_numpy=True)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(EMBED_FILE))

    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    # update in-memory globals so UI sees changes without restart
    global _INDEX, _CHUNKS
    _INDEX = index
    _CHUNKS = chunks
    logger.info("Ingest complete: saved %d chunks", len(chunks))


def ask(query: str, k: int = 3) -> str:
    if not query:
        return ""
    index, chunks = _load_index_and_chunks()
    model = _load_model()
    q_vec = model.encode([query], convert_to_numpy=True)
    if q_vec.dtype != np.float32:
        q_vec = q_vec.astype(np.float32)
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)
    D, I = index.search(q_vec, k)
    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return "\n\n".join(results)

def search(query: str, k: int = 3) -> str:
    """Compatibility wrapper so app.py can import `search` (calls ask)."""
    return ask(query, k)

