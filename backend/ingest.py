import argparse
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
NOTES_DIR = BASE_DIR / "notes"
VECTOR_DIR = BASE_DIR / "vector_store"
EMBED_FILE = VECTOR_DIR / "embeddings.faiss"
META_FILE = VECTOR_DIR / "metadata.pkl"


def setup_directories() -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    if not NOTES_DIR.exists():
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Created notes/ folder. Put PDFs there or pass --pdf to ingest a single file.")


def read_pdfs(single_pdf: Path | None = None) -> str:
    text_data: list[str] = []
    if single_pdf:
        pdf_paths = [single_pdf]
    else:
        pdf_paths = sorted(NOTES_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError("❌ No PDF files found. Put PDFs in backend/notes/ or pass --pdf <path>")

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            page_texts: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    page_texts.append(text)
            joined = " ".join(page_texts).strip()
            if joined:
                text_data.append(joined)
            logger.info("✅ Loaded: %s", pdf_path.name)
        except Exception as e:
            logger.error("⚠️ Error reading %s: %s", pdf_path, e)

    return "\n\n".join(text_data)


def _simple_split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 120) -> list[str]:
    chunks: list[str] = []
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

def chunk_text(text: str) -> list[str]:
    """Try LangChain splitter first, otherwise use local fallback."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        return splitter.split_text(text)
    except Exception:
        logger.warning("langchain splitter unavailable — using simple fallback splitter")
        return _simple_split_text(text, chunk_size=600, chunk_overlap=120)


def build_vector_db(single_pdf: Path | None = None) -> None:
    try:
        setup_directories()

        logger.info("📄 Reading PDFs...")
        text = read_pdfs(single_pdf)

        if not text.strip():
            raise ValueError("No text extracted from PDFs.")

        logger.info("✂️ Splitting text into chunks...")
        chunks = chunk_text(text)
        logger.info("Total chunks created: %d", len(chunks))

        logger.info("⚙️ Generating embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

        # Ensure embeddings are float32 for FAISS
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        logger.info("💾 Saving FAISS DB...")
        faiss.write_index(index, str(EMBED_FILE))

        with open(META_FILE, "wb") as f:
            pickle.dump(chunks, f)

        logger.info("✅ Vector Database saved in: %s", VECTOR_DIR)

    except Exception as e:
        logger.error("❌ Error building vector DB: %s", e)
        raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest PDFs into FAISS vector DB")
    p.add_argument("--pdf", "-p", help="Single PDF path to ingest (optional)", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pdf_path = Path(args.pdf) if args.pdf else None
    if pdf_path and not pdf_path.exists():
        raise FileNotFoundError(f"Provided PDF not found: {pdf_path}")
    build_vector_db(pdf_path)


