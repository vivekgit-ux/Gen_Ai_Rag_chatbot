import os
import sys
import warnings
import logging

# 1. Suppress all Python warnings (DeprecationWarning, UserWarning, etc.)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 2. Prevent duplicate API key warning from Google SDK
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

# 3. Suppress verbose library loggers
for logger_name in [
    "google_genai",
    "google_genai._api_client",
    "google_genai.models",
    "httpx",
    "flashrank",
    "flashrank.Ranker",
    "qdrant_client",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from retrieval.retriever import Retriever
from generation.llm import LLMGenerator


class ConversationalRAG:
    def __init__(self, collection_name: str = "brics_rag", top_k: int = 3):
        self.retriever = Retriever(top_k=top_k, collection_name=collection_name)
        self.generator = LLMGenerator()
        self.chat_history = []

    def chat(self, user_query: str) -> str:
        # Rewrite query if conversation history exists
        if self.chat_history:
            search_query = self.generator.rewrite_query(user_query, self.chat_history)
            print(f"\n[Rewritten Query]: '{user_query}' -> '{search_query}'")
        else:
            search_query = user_query

        # Retrieve relevant contexts
        retrieved_docs = self.retriever.retrieve(search_query)
        context = self.retriever.format_context(retrieved_docs) if retrieved_docs else "No context found."

        # Generate response with citations
        answer = self.generator.generate_response(
            query=user_query,
            context=context,
            chat_history=self.chat_history
        )

        # Update chat history
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer


def main():
    print("\n" + "=" * 55)
    print("      BRICS Statistical Report Conversational RAG")
    print("=" * 55 + "\n")

    rag = ConversationalRAG(collection_name="brics_rag", top_k=3)

    test_queries = [
        "What was China's coal production in 2025 according to the report?",
        "How does that compare to India for the same year?",
        "What is the projected commercial drone market revenue in South Africa?"
    ]

    for q in test_queries:
        print(f"\nUser: {q}")
        response = rag.chat(q)
        print(f"\nAssistant:\n{response}\n")
        print("-" * 55)


if __name__ == "__main__":
    main()