# 🤖 YILGPT RAG Pipeline Overview

The YILGPT system follows a standard Retrieval-Augmented Generation (RAG) architecture.

## RAG Flow 
1.  **Document Ingestion:**
    * PDF files are loaded and parsed into raw text pages (`ingestion.py`).
    * Text is cleaned (`utils/text_cleaning.py`).
    * Cleaned text is split into small, overlapping **chunks** (`ingestion.py`).
    * Metadata (e.g., `document_type`: HR, IT, OPS) is assigned to each chunk based on the source filename.

2.  **Indexing (VectorDB Creation):**
    * Each text chunk is converted into a high-dimensional vector **embedding** using a `SentenceTransformer` model (`vector_db.py`).
    * These embeddings and their corresponding chunks are stored in the in-memory `VectorDB`.

3.  **Query Time (Retrieval & Generation):**
    * **Intent Detection:** The user's question is analyzed to determine the document type (e.g., "HR" for compliance questions) for filtering (`retrieval.py`).
    * **Embedding & Search:** The question is embedded, and the `VectorDB` searches for the most similar chunk embeddings, filtering by the detected intent (`vector_db.py`). The top K (default 3) relevant chunks are retrieved.
    * **Prompt Construction:** The original question and the retrieved text chunks are combined into a single, structured prompt (`llm_client.py`).
    * **Generation:** This final prompt is sent to the LLM (mocked by `LLMClient`), which generates a final answer **based only on the provided context**.