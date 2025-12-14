import os

# --- Embedding & LLM configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
CONTEXT_WINDOW = 4096
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10
TOP_K = 3 # Used for retrieval in the final pipeline

# --- Vector DB configuration ---
VECTOR_DB_PATH = "Training_pdfs/vdb"
INDEX_NAME = "yilgpt_index"

# --- Document paths (Not fully used in the notebook, but included for completeness) ---
DATA_ROOT = "Training_pdfs"
MANUALS_DIR = f"{DATA_ROOT}/manuals"
SOPS_DIR = f"{DATA_ROOT}/sops"
LOGS_DIR = f"{DATA_ROOT}/alarm_logs"

# --- Loader Configuration ---
PDF_FOLDER = "Training_pdfs"