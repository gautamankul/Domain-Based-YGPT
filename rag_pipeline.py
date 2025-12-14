from vector_db import VectorDB
from llm_client import LLMClient
from retrieval import Retriever


class YILGPT:
    """
    The main RAG pipeline class, encapsulating retrieval and generation.
    """

    def __init__(self, vector_db: VectorDB):
        self.vdb = vector_db
        # Initialize the components
        self.llm = LLMClient()
        self.retriever = Retriever(self.vdb, self.llm)

    def query(self, question: str) -> str:
        """
        Executes the full RAG pipeline:
        1. Retrieve relevant documents based on the question and intent.
        2. Generate an answer using the retrieved documents as context.
        """
        # 1. Retrieval
        docs = self.retriever.retrieve(question)

        # 2. Answer Generation (returning the prompt template as per the notebook)
        return self.llm.generate_answer(question, docs)