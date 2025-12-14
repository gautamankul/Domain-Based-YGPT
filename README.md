# YILGPT: Document-Specific RAG Chatbot

YILGPT is a Retrieval-Augmented Generation (RAG) system designed to answer questions based on a collection of internal documents (HR policies, IT SOPs, OPS logs, etc.).

This project is structured based on a modular design, separating key components like ingestion, vector storage, retrieval, and LLM interaction. The core logic is exposed via a **FastAPI** service.

## Project Structure Overview

* `main.py`: The entry point for the FastAPI application.
* `config.py`: Stores all configuration constants (model names, chunk sizes, paths).
* `ingestion.py`: Handles loading documents (using `PyPDFLoader`) and chunking the text.
* `vector_db.py`: Implements the in-memory vector database (`VectorDB`) and embedding logic (`SentenceTransformer`).
* `llm_client.py`: Mocks the interaction with a Large Language Model.
* `retrieval.py`: Manages the document retrieval process, including intent detection and metadata filtering.
* `rag_pipeline.py`: The orchestrator class (`YILGPT`).
* `api.py`: Contains the initial loading and setup logic for the RAG pipeline.
* `utils/`: Helper functions for text cleaning, logging, and file reading.

## 🚀 Setup and Running the API

### 1. Prerequisites

You must have Python 3.8+ installed.

### 2. Install Dependencies

Install all required Python libraries, including those used for the RAG pipeline and the FastAPI server.

```bash
# Install core RAG dependencies (based on the original notebook setup)
pip install numpy sentence-transformers pypdf langchain-community 

# Install FastAPI and Uvicorn (the ASGI server)
pip install fastapi "uvicorn[standard]"
```


### 2. Run Application

* Prepare Documents
Create a folder named Training_pdfs in your environment (or wherever your PDF_FOLDER constant points in yil_gpt/config.py).

* Place your training PDF documents (e.g., company_policy.pdf, ai_basics.pdf) inside the Training_pdfs folder.

### Run the Server
Start the FastAPI application using Uvicorn. The RAG pipeline will be initialized upon startup.

```bash
uvicorn main:app --reload

main:app: Specifies to run the FastAPI application named app inside the main.py file.

--reload: (Optional, recommended for development) Automatically restarts the server when code changes are detected.
```


### Access the API
Once the server is running, the API will typically be available at http://127.0.0.1:8000.

Interactive Docs: Open your browser to http://127.0.0.1:8000/docs to use the integrated Swagger UI for testing.

Query Endpoint: Send a POST request to the /query endpoint.

Example cURL Request:
```aiignore
curl -X 'POST' \
  '[http://127.0.0.1:8000/query](http://127.0.0.1:8000/query)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What does the HR policy say about compliance?"
}'
```
