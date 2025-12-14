from typing import Dict, List, Optional
from vector_db import VectorDB
from llm_client import LLMClient
from config import TOP_K  # Use TOP_K from config


def detect_intent(query: str) -> str:
    """
    Classifies the intent of the query based on keywords to facilitate
    metadata-based filtering (Document Type).
    """
    q = query.lower()
    if any(k in q for k in ["policy", "leave", "compliance", "hr"]):
        return "HR"
    if any(k in q for k in ["incident", "outage", "failure"]):
        return "IT"
    if any(k in q for k in ["alarm", "compressor", "pump"]):
        return "OPS"
    return "GENERAL"


class Retriever:
    """Handles the retrieval step of the RAG pipeline."""

    def __init__(self, vdb: VectorDB, llm: LLMClient):
        self.vdb = vdb
        self.llm = llm

    def retrieve(self, query: str) -> List[Dict]:
        """
        Performs query rewriting, intent detection, and vector search.
        """
        # 1. Query Rewriting/Enhancement
        rewritten_query = self.llm.rewrite_query(query)

        # 2. Intent Detection for Filtering
        intent = detect_intent(rewritten_query)

        # 3. Vector Search with Metadata Filtering
        retrieved_docs = self.vdb.search(
            rewritten_query,
            top_k=TOP_K,
            filters={"document_type": intent}
        )

        return retrieved_docs