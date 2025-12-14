import os
from ingestion import load_documents_from_folder, process_documents
from vector_db import VectorDB
from rag_pipeline import YILGPT
from config import PDF_FOLDER
from utils.logger import get_logger

logger = get_logger()

# Global variables for the bot instance and database
global_vdb: VectorDB = None
global_bot: YILGPT = None


def initialize_bot() -> YILGPT:
    """
    Initializes and loads the VectorDB and the YILGPT RAG bot.
    This function recreates the setup steps from the notebook.
    """
    global global_vdb, global_bot

    if global_bot is not None:
        logger.info("YILGPT bot is already initialized.")
        return global_bot

    logger.info("Starting bot initialization...")

    # --- 1. Load Documents ---
    # NOTE: The notebook used a hardcoded path '/content/Training_pdfs'. 
    # Ensure this path exists and contains PDFs if running this locally.
    lc_documents = load_documents_from_folder(PDF_FOLDER)
    if not lc_documents:
        logger.warning("No documents loaded. Bot will have limited functionality.")

    # --- 2. Process and Chunk ---
    all_chunks = process_documents(lc_documents)
    logger.info("Total chunks created: %d", len(all_chunks))

    # --- 3. Build Vector Database ---
    global_vdb = VectorDB()
    global_vdb.add_documents(all_chunks)

    # --- 4. Initialize RAG Pipeline ---
    global_bot = YILGPT(global_vdb)
    logger.info("YILGPT bot initialization complete.")

    return global_bot


def query_bot(question: str) -> str:
    """
    The main public interface to query the RAG bot.
    """
    if global_bot is None:
        bot = initialize_bot()
    else:
        bot = global_bot

    try:
        response = bot.query(question)
        return response
    except Exception as e:
        logger.error("Error during query: %s", e)
        return f"An error occurred while processing your request: {e}"


# Example of how to run the final test from the notebook:
if __name__ == "__main__":
    bot_instance = initialize_bot()

    # Test case 1 from the notebook
    print("\n--- Test 1: HR Compliance ---")
    question_hr = "What does the document say about compliance?"
    answer_hr = bot_instance.query(question_hr)
    print(answer_hr)

    # Test case 2: OPS
    print("\n--- Test 2: OPS Alarm ---")
    question_ops = "What happened to Compressor C-101?"
    answer_ops = bot_instance.query(question_ops)
    print(answer_ops)

    # Test case 3: GENERAL
    print("\n--- Test 3: General AI Topic ---")
    question_general = "Artificial Intelligence (AI) refers to systems that can perform tasks requiring human intelligence."
    answer_general = bot_instance.query(question_general)
    print(answer_general)