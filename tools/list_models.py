# Run: python tools\list_models.py
from dotenv import load_dotenv
load_dotenv()
import os, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except Exception as e:
    logger.error("generativeai client import error: %s", e)
    raise

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

genai.configure(api_key=key)

# list models
try:
    models = genai.list_models()
    logger.info("Models: %s", getattr(models, "models", models))
except Exception as e:
    logger.error("list_models failed: %s", e)

# try a quick generate (pick a model from the list above if needed)
try:
    # change model_name to one from the printed list if "gemini-pro" fails
    model_name = "gemini-pro"
    resp = genai.GenerativeModel(model_name).generate_content("Say hello in one short sentence.")
    logger.info("Resp: %s", resp)
except Exception as e:
    logger.error("generate_content failed: %s", e)