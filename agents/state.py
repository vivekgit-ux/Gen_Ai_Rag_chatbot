from typing import List, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
  question: str
  file_path: Optional[str]
  generation: str
  route: Optional[str]  # "ingest", "retrieve", "direct_chat", "blocked"
  documents: List[dict]  
  documents_relevant: bool
  retry_count: int
  messages: Sequence[BaseMessage]