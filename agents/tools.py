import os
import sys
from langchain_core.tools import tool

# Ensure project root is visible to imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retriever import Retriever

# Singleton retriever instance
_retriever_instance = None


def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever(top_k=3)
    return _retriever_instance


@tool
def search_knowledge_base(query: str) -> str:
    """Useful when you need to answer questions based on indexed internal documents,
    reports, tables, or private knowledge base data.
    Input should be a clean, targeted search query string.
    """
    try:
        retriever = get_retriever()
        docs = retriever.retrieve(query)
        if not docs:
            return "No relevant documents found in the internal knowledge base."
        return retriever.format_context(docs)
    except Exception as e:
        return f"Retrieval error occurred: {str(e)}"