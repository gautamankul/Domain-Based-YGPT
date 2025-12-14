# YILGPT: Document-Specific RAG Chatbot

YILGPT is a Retrieval-Augmented Generation (RAG) system designed to answer questions based on a collection of internal documents (HR policies, IT SOPs, OPS logs, etc.).

This project is structured based on a modular design, separating key components like ingestion, vector storage, retrieval, and LLM interaction.

## Project Structure Overview

* `config.py`: Stores all configuration constants (model names, chunk sizes, paths).
* `ingestion.py`: Handles loading documents (using `PyPDFLoader`) and chunking the text.
* `vector_db.py`: Implements the in-memory vector database (`VectorDB`) and embedding logic (`SentenceTransformer`).
* `llm_client.py`: Mocks the interaction with a Large Language Model for query rewriting and answer generation.
* `retrieval.py`: Manages the document retrieval process, including query intent detection and metadata filtering.
* `rag_pipeline.py`: The orchestrator class (`YILGPT`) that ties the retrieval and LLM components together.
* `api.py`: Contains the initialization logic and the main public interface (`query_bot`) to interact with the system.
* `utils/`: Helper functions for text cleaning, logging, and file reading.

## Setup and Run

1.  **Dependencies:** Ensure all required libraries from the notebook are installed (`langchain-community`, `transformers`, `sentence-transformers`, `pypdf`, `numpy`).
2.  **PDFs:** Place your training PDF documents in the directory specified by `PDF_FOLDER` in `config.py` (e.g., `/content/Training_pdfs`).
3.  **Run:** Execute `python -m yil_gpt.api` to initialize the bot and run the test cases.