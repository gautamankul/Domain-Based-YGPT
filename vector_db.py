from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger()

# Initialize the embedding model outside the class for efficiency
# This assumes the user has run the setup or the environment is configured.
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
except Exception as e:
    logger.error("Failed to load SentenceTransformer model %s: %s", EMBEDDING_MODEL, e)


    # Define a dummy function if loading fails to allow class definition
    def embed_text(text: str) -> np.ndarray:
        return np.zeros(384, dtype="float32")  # all-MiniLM-L6-v2 uses 384 dimensions


def embed_text(text: str) -> np.ndarray:
    """Encodes text into a normalized vector embedding."""
    return embedding_model.encode(text, normalize_embeddings=True).astype("float32")


@dataclass
class VectorDB:
    """
    A minimal, in-memory vector database using cosine similarity 
    (via dot product of normalized vectors).
    """
    vectors: List[np.ndarray] = field(default_factory=list, repr=False)
    docs: List[Dict[str, Any]] = field(default_factory=list)

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Embeds and stores documents in the database."""
        for doc in docs:
            # The chunk's text is already cleaned/chunked at this point
            self.vectors.append(embed_text(doc["text"]))
            self.docs.append(doc)
        logger.info("Added %d documents/vectors to the database.", len(docs))

    def search(
            self,
            query: str,
            top_k: int,
            filters: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Performs vector search with optional metadata filtering.
        Uses dot product on normalized vectors for cosine similarity.
        """
        if not self.vectors:
            return []

        q_vec = embed_text(query)
        sims = []

        for idx, doc in enumerate(self.docs):
            # --- Metadata filtering ---
            is_match = True
            if filters:
                for k, v in filters.items():
                    if doc["metadata"].get(k) != v:
                        is_match = False
                        break

            if not is_match:
                continue

            # Calculate cosine similarity (dot product of normalized vectors)
            score = np.dot(self.vectors[idx], q_vec)
            sims.append((idx, score))

        # Sort by similarity score (descending)
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[:top_k]

        # Return the retrieved documents with their scores
        return [
            self.docs[i] | {"score": float(score)}
            for i, score in sims
        ]
