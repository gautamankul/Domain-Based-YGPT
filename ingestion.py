import os
from typing import Dict, List
from langchain_community.document_loaders import PyPDFLoader
from config import PDF_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger
from utils.text_cleaning import clean_text

logger = get_logger()


def load_documents_from_folder(folder_path: str = PDF_FOLDER) -> List[Dict]:
    """Loads all PDF files from a specified folder using PyPDFLoader and returns a list of LangChain documents."""
    documents = []
    logger.info("Starting document loading from: %s", folder_path)
    if not os.path.isdir(folder_path):
        logger.error("Folder not found: %s", folder_path)
        return []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            try:
                # Use LangChain's PyPDFLoader for robust PDF loading
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                logger.info("Loaded %d pages from %s", len(documents), file)
            except Exception as e:
                logger.error("Error loading PDF %s: %s", file, e)

    return documents


def get_document_type(source: str) -> str:
    """Determines the document type based on keywords in the source path."""
    source_lower = source.lower()
    if "policy" in source_lower or "company" in source_lower or "hr" in source_lower:
        return "HR"
    elif "incident" in source_lower or "sop" in source_lower:
        return "IT"
    elif "alarm" in source_lower or "log" in source_lower:
        return "OPS"
    else:
        return "GENERAL"


def create_chunks(text: str, metadata: Dict) -> List[Dict]:
    """Splits text into overlapping word-based chunks with metadata."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_text = " ".join(words[start:end])
        if not chunk_text.strip():
            break

        chunk_metadata = metadata.copy()

        # Add document_type based on the source metadata
        source = chunk_metadata.get("source", "")
        chunk_metadata["document_type"] = get_document_type(source)

        chunks.append({
            "text": chunk_text,
            "metadata": chunk_metadata
        })
        # Calculate the next starting point with overlap
        start += max(CHUNK_SIZE - CHUNK_OVERLAP, 1)

    logger.info("Created %d chunks", len(chunks))
    return chunks


def process_documents(lc_documents: List) -> List[Dict]:
    """Cleans raw document content and splits it into chunks."""
    all_chunks = []
    for doc in lc_documents:
        raw_text = doc.page_content
        metadata = doc.metadata.copy()

        # Clean the text before chunking
        cleaned_text = clean_text(raw_text)

        doc_chunks = create_chunks(cleaned_text, metadata)
        all_chunks.extend(doc_chunks)

    return all_chunks
