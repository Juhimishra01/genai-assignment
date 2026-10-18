import os
from dotenv import load_dotenv
load_dotenv()
def _i(k, d): return int(os.getenv(k, d))
def _f(k, d): return float(os.getenv(k, d))

CHUNK_SIZE = _i("CHUNK_SIZE", 800)        # characters (default)
CHUNK_OVERLAP = _i("CHUNK_OVERLAP", 120)  # characters (default)
TOP_K = _i("TOP_K", 5)
MIN_SIM = _f("MIN_SIM", 0.30)             # cosine similarity gate for "no relevant context"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("COLLECTION", "docs")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EVAL_LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL") or LLM_BASE_URL
EVAL_LLM_API_KEY = os.getenv("EVAL_LLM_API_KEY") or LLM_API_KEY
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL") or LLM_MODEL
LOG_PATH = os.getenv("LOG_PATH", "logs/queries.jsonl")
