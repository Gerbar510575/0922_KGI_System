import os

# ---------------- Paths ----------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma_db")
DB_NAME    = os.getenv("DB_NAME", "funds")

# ---------------- Models ----------------
GEMINI_MODEL    = "gemini-2.5-flash"
GENAI_API_KEY   = os.getenv("GENAI_API_KEY", "")

# ---------------- Retrieval Defaults ----------------
TOP_K     = 5
SHOW_CHAR = 200




