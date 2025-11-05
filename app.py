import os
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from backend.rag_engine import ingest_pdf, search
from backend.geminiapi import gemini_answer, generate_summary, generate_mindmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def _has_key() -> bool:
    return bool(GEMINI_API_KEY)

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    _GENAI_AVAILABLE = True
except Exception as e:
    logger.warning("google.generativeai not available: %s", e)
    _GENAI_AVAILABLE = False

def _call_genai(prompt: str) -> str:
    if not _GENAI_AVAILABLE or not _has_key():
        raise RuntimeError("Generative AI client not available or GEMINI_API_KEY missing")
    try:
        # call the library but guard against model NotFound etc.
        resp = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        # log full exception and raise to caller to handle fallback
        logger.exception("Generative API call failed: %s", e)
        raise

def gemini_answer(question: str, context: Optional[str]) -> str:
    prompt = f"""You are a RAG assistant. Answer based only on context.

Context:
{context or ""}

Question: {question}
"""
    try:
        return _call_genai(prompt)
    except Exception:
        # fallback stub
        excerpt = (context or "")[:800].replace("\n", " ")
        return f"(stub) Unable to call Gemini API. Returning context excerpt:\n\n{excerpt}"

def generate_summary(context: Optional[str]) -> str:
    prompt = f"Create a very short bullet summary:\n\n{context or ''}"
    try:
        return _call_genai(prompt)
    except Exception:
        if not context:
            return "No context available to summarize."
        lines = [ln.strip() for ln in (context or "").splitlines() if ln.strip()]
        return "\n".join(f"• {ln}" for ln in lines[:10])

def generate_mindmap(context: Optional[str]) -> str:
    prompt = f"""Create a mind-map in text:

Topic
 ├─ Main point
 │   ├─ Sub point

Text:
{context or ""}
"""
    try:
        return _call_genai(prompt)
    except Exception:
        if not context:
            return "No context available to generate mind map."
        items = [ln.strip() for ln in (context or "").splitlines() if ln.strip()][:12]
        return "MindMap (stub):\n" + "\n".join(f"- {it}" for it in items)

st.title("📚 Smart PDF Gemini RAG Notebook")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded is not None:
    st.write(f"Selected file: {uploaded.name} ({uploaded.size} bytes)")
    if st.button("Process PDF"):
        try:
            file_path = UPLOAD_DIR / uploaded.name
            # Save uploaded file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.info(f"Saved {uploaded.name} → {file_path}")
            with st.spinner("Ingesting PDF into vector DB..."):
                ingest_pdf(file_path)
            st.success("Ingest complete.")
        except Exception as e:
            logger.exception("Failed to ingest PDF")
            st.error(f"Ingest failed: {e}")

st.divider()

query = st.text_input("Ask question from PDF", value="")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                context = search(query, k=3)
                answer = gemini_answer(query, context)
            st.write("### ✅ Answer")
            st.write(answer)
            st.write("### 🔎 Retrieved context (excerpt)")
            st.code((context or "")[:2000])
        except FileNotFoundError:
            st.error("Vector DB not found. Please upload and process a PDF first.")
        except Exception as e:
            logger.exception("Search/Gemini failed")
            st.error(f"Error: {e}")

if st.button("Generate Summary"):
    try:
        with st.spinner("Generating summary..."):
            # use the query to focus summary if provided, otherwise get broad context
            context = search(query or "summary", k=5)
            summary = generate_summary(context)
        st.write("### 📝 Summary")
        st.write(summary)
    except FileNotFoundError:
        st.error("Vector DB not found. Please upload and process a PDF first.")
    except Exception as e:
        logger.exception("Generate summary failed")
        st.error(f"Error generating summary: {e}")

if st.button("Generate Mindmap"):
    try:
        with st.spinner("Generating mind map..."):
            context = search(query or "mindmap", k=6)
            mindmap = generate_mindmap(context)
        st.write("### 🧠 Mind Map")
        st.code(mindmap)
    except FileNotFoundError:
        st.error("Vector DB not found. Please upload and process a PDF first.")
    except Exception as e:
        logger.exception("Generate mindmap failed")
        st.error(f"Error generating mind map: {e}")
