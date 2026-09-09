import os
import warnings
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

for name in ["google_genai", "google_genai.models", "httpx"]:
    logging.getLogger(name).setLevel(logging.ERROR)

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


def _extract_text(content: Any) -> str:
    """Safely extracts text regardless of whether response.content is str or list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(str(getattr(part, "text")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


class LLMGenerator:
    def __init__(self, model_name: str = "gemini-3.5-flash-lite", temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=5
        )

    def rewrite_query(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """Rewrites a conversational follow-up into a standalone search query."""
        if not chat_history:
            return question

        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])

        prompt = (
            "Given the following conversation history and a follow-up question, rewrite the follow-up "
            "question into an independent, standalone search query.\n\n"
            "CRITICAL RULES:\n"
            "- Retain all entities, country names, indicators, and years from previous questions.\n"
            "- If the question asks for a comparison, INCLUDE BOTH subjects (e.g., 'China and India coal production 2025 comparison').\n"
            "- Return ONLY the rewritten query text without any markdown or quotes.\n\n"
            f"Chat History:\n{history_str}\n\n"
            f"Follow-up Question: {question}\n"
            "Standalone Query:"
        )

        response = self.llm.invoke(prompt)
        text = _extract_text(response.content)
        return text.strip().strip('"').strip("'")

    def generate_response(self, query: str, context: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generates grounded responses with inline citations and references."""
        system_instruction = (
            "You are an expert statistical research analyst answering strictly using the provided context.\n\n"
            "RULES:\n"
            "1. Answer ONLY using the facts, numbers, and tables explicitly stated in the context.\n"
            "2. Whenever you state any metric or claim, attach an inline citation indicating source and page "
            "immediately after the clause (e.g., '[Source 1, Page 42]').\n"
            "3. If the context does not contain enough information, state clearly that the provided documents "
            "do not contain this data. Do not invent or assume numbers.\n"
            "4. At the end of your answer, include a '### References & Citations' list detailing each source, "
            "its page number, and the extracted quote or snippet."
        )

        history_str = ""
        if chat_history:
            history_str = "Conversation History:\n" + "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]]
            ) + "\n\n"

        user_content = (
            f"{history_str}"
            f"Context Documents:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Grounded Answer with In-Text Citations:"
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content)
        ]

        response = self.llm.invoke(messages)
        text = _extract_text(response.content)
        return text.strip()