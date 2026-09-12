import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agents.nodes import (
    auto_ingestion_node,
    direct_chat_node,
    generate_node,
    grade_documents_node,
    hallucination_grader_node,
    rerank_node,
    retrieve_node,
    rewrite_query_node,
    triage_node,
)
from agents.state import AgentState


def triage_edge(state: AgentState) -> str:
  route = state.get("route")
  if route == "ingest":
    return "auto_ingestion"
  elif route == "retrieve":
    return "retrieve"
  elif route == "direct_chat":
    return "direct_chat"
  return "end"


def relevance_edge(state: AgentState) -> str:
  if state.get("documents_relevant", False):
    return "generate"
  if state.get("retry_count", 0) < 2:
    return "rewrite_query"
  return "generate"


workflow = StateGraph(AgentState)

workflow.add_node("triage", triage_node)
workflow.add_node("auto_ingestion", auto_ingestion_node)
workflow.add_node("direct_chat", direct_chat_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("hallucination_grader", hallucination_grader_node)

workflow.set_entry_point("triage")

workflow.add_conditional_edges(
    "triage",
    triage_edge,
    {
        "auto_ingestion": "auto_ingestion",
        "retrieve": "retrieve",
        "direct_chat": "direct_chat",
        "end": END,
    },
)

workflow.add_edge("auto_ingestion", END)
workflow.add_edge("direct_chat", END)

workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    relevance_edge,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
    },
)
workflow.add_edge("rewrite_query", "retrieve")

workflow.add_edge("generate", "hallucination_grader")
workflow.add_edge("hallucination_grader", END)

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

app_graph = workflow.compile(checkpointer=memory)