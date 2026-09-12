import uuid
from typing import Any, List, Optional, cast
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.graph import app_graph

app = FastAPI(
    title="Enterprise Agentic RAG API",
    description="Production-grade LangGraph Agent with Hybrid Search, Self-Correction, and Persistent Memory.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
  question: str = Field(
      ...,
      min_length=1,
      json_schema_extra={
        "example": "What are the key findings in the documents?",
      },
  )
  thread_id: Optional[str] = Field(
      default=None,
      description="Persistent conversation session ID. If omitted, a new UUID is generated.",
  )


class DocumentMetadata(BaseModel):
  source: str
  page: int
  chunk_id: str


class DocumentChunk(BaseModel):
  content: str
  metadata: DocumentMetadata


class ChatResponse(BaseModel):
  thread_id: str
  answer: str
  route_taken: Optional[str] = None
  documents: Optional[List[DocumentChunk]] = []


def normalize_content(content) -> str:
  """Normalizes Gemini string, dict, or list-of-dicts content into a single plain string."""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    chunks = []
    for item in content:
      if isinstance(item, dict):
        chunks.append(item.get("text", ""))
      elif hasattr(item, "text"):
        chunks.append(getattr(item, "text", ""))
      else:
        chunks.append(str(item))
    return "".join(chunks)
  return str(content)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
  """Health check endpoint for container orchestrators and monitoring tools."""
  return {"status": "healthy", "service": "enterprise-agentic-rag"}


@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(request: ChatRequest):
  """Executes a full turn across the LangGraph Agentic pipeline."""
  session_id = request.thread_id or str(uuid.uuid4())
  config: RunnableConfig = {"configurable": {"thread_id": session_id}}

  try:
    result = app_graph.invoke(
        cast(Any, {
            "question": request.question,
            "messages": [HumanMessage(content=request.question)],
            "retry_count": 0,
        }),
        config=config,
    )

    raw_generation = result.get("generation", "No response generated.")
    answer_str = normalize_content(raw_generation)

    docs_payload = []
    for doc in result.get("documents", []):
      doc_content = normalize_content(doc.get("content", ""))
      doc_meta = doc.get("metadata", {})
      docs_payload.append(
          DocumentChunk(
              content=doc_content,
              metadata=DocumentMetadata(
                  source=str(doc_meta.get("source", "Internal Document")),
                  page=int(doc_meta.get("page", 1)),
                  chunk_id=str(doc_meta.get("chunk_id", "N/A")),
              ),
          )
      )

    return ChatResponse(
        thread_id=session_id,
        answer=answer_str,
        route_taken=result.get("route", "direct_chat"),
        documents=docs_payload,
    )

  except Exception as exc:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Agent workflow failed: {str(exc)}",
    )