from typing import List, Dict

class LLMClient:
    """
    A mock class to simulate LLM interaction for query rewriting and answer generation.
    The real implementation would involve an API call to a model like gpt-4o-mini.
    """

    def rewrite_query(self, query: str) -> str:
        """
        Simulates query rewriting/enhancement (e.g., for RAG optimization).
        As per the notebook, this is a simple pass-through.
        """
        # A more advanced RAG would use an LLM call here:
        # rewritten_query = LLM_API.call(prompt="Rewrite this query:...")
        return query.strip()

    def generate_answer(self, query: str, docs: List[Dict]) -> str:
        """
        Generates a final answer using the retrieved documents as context.
        This method returns the prompt that would be sent to the LLM.
        """
        if not docs:
            return "Answer not available in the selected document."

        # Format context for the LLM prompt
        context = "\n\n".join(
            f"[{d['metadata'].get('document_type', 'UNKNOWN')}] {d['text']}"
            for d in docs
        )

        # Construct the RAG prompt
        return f"""
Answer ONLY using the context below.

Question:
{query}

Context:
{context}
"""