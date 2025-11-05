import os
import logging
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USE_GENAI = os.getenv("USE_GENAI", "1").strip() not in ("0", "false", "False", "no", "NO")

_genai = None
_genai_initialized = False
_model_cache: list[str] = []

def _init_genai():
    global _genai, _genai_initialized
    if _genai_initialized:
        return _genai
    _genai_initialized = True
    if not USE_GENAI or not GEMINI_API_KEY:
        logger.info("Generative AI disabled or key missing.")
        return None
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=GEMINI_API_KEY)
        _genai = genai
        logger.info("Generative AI client configured (lazy).")
        return _genai
    except Exception as e:
        logger.warning("Failed to initialize generativeai client: %s", e)
        _genai = None
        return None

def _discover_models(genai_client) -> list[str]:
    global _model_cache
    if _model_cache:
        return _model_cache
    try:
        items = genai_client.list_models()
        models = list(items) if items is not None else []
        names = []
        for m in models:
            name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
            if name:
                names.append(name)
        _model_cache = names
        logger.info("Discovered models: %s", names)
        return names
    except Exception as e:
        logger.debug("list_models failed: %s", e)
        return []

def _call_genai(prompt: str) -> str:
    genai = _init_genai()
    if not genai:
        raise RuntimeError("Generative AI disabled or not configured")

    tried = []
    # get candidate model names
    candidates = _discover_models(genai) + ["models/text-bison-001", "gemini-pro", "gemini-1.0"]
    for model_name in candidates:
        if not model_name or model_name in tried:
            continue
        tried.append(model_name)
        try:
            # try generate_text (preferred if available)
            if hasattr(genai, "generate_text"):
                out = genai.generate_text(model=model_name, prompt=prompt)
                return getattr(out, "text", str(out))
            # fallback to GenerativeModel API
            if hasattr(genai, "GenerativeModel"):
                resp = genai.GenerativeModel(model_name).generate_content(prompt)
                if hasattr(resp, "candidates") and resp.candidates:
                    try:
                        return resp.candidates[0].content[0].text
                    except Exception:
                        return str(resp)
                if hasattr(resp, "text"):
                    return resp.text
                return str(resp)
        except Exception as e:
            logger.debug("Model %s failed: %s", model_name, e)
            continue

    raise RuntimeError("No working generative model found")

def gemini_answer(question: str, context: Optional[str]) -> str:
    prompt = f"You are a RAG assistant. Answer only from the context.\n\nContext:\n{context or ''}\n\nQuestion: {question}"
    try:
        return _call_genai(prompt)
    except Exception as e:
        logger.warning("GenAI failed: %s -- returning stub", e)
        excerpt = (context or "")[:1000].replace("\n", " ")
        return f"(stub) Generative API unavailable. Context excerpt:\n\n{excerpt}"

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
    prompt = f"Create a mind-map in text format from the following content:\n\n{context or ''}"
    try:
        return _call_genai(prompt)
    except Exception:
        if not context:
            return "No context available to generate mind map."
        items = [ln.strip() for ln in (context or "").splitlines() if ln.strip()][:12]
        return "MindMap (stub):\n" + "\n".join(f"- {it}" for it in items)
