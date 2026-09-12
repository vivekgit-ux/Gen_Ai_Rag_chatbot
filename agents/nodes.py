import os
import uuid
from typing import cast
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from agents.guards import HallucinationScore, RelevanceScore, TriageOutput
from agents.state import AgentState

load_dotenv(override=True)


def extract_text_safely(content) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for item in content:
      if isinstance(item, dict):
        parts.append(item.get("text", ""))
      elif hasattr(item, "text"):
        parts.append(getattr(item, "text", ""))
      else:
        parts.append(str(item))
    return "".join(parts)
  return str(content)


# 1. Models (temperature parameter removed to avoid warnings)
eval_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_retries=3)
gen_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_retries=3)

# 2. Qdrant Setup
COLLECTION_NAME = "enterprise_knowledge"
EMBEDDING_DIM = 3072

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
qdrant_client = QdrantClient(path="qdrant_db")

if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
  qdrant_client.create_collection(
      collection_name=COLLECTION_NAME,
      vectors_config=qmodels.VectorParams(
          size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE
      ),
      sparse_vectors_config={
          "sparse": qmodels.SparseVectorParams(
              index=qmodels.SparseIndexParams(on_disk=False)
          )
      },
  )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    sparse_vector_name="sparse",
    retrieval_mode=RetrievalMode.HYBRID,
)

# Initial fetch of 6 chunks; reranker will pick top 3
retriever = vector_store.as_retriever(search_kwargs={"k": 6})


# -------------------------------------------------------------
# Nodes
# -------------------------------------------------------------


def triage_node(state: AgentState) -> dict:
  question = state.get("question", "")
  structured_triage = eval_llm.with_structured_output(TriageOutput)

  system_prompt = (
      "Analyze the request:\n1. If user asks to load/index/ingest a file, set"
      " route='ingest' and extract file path.\n2. If user asks a domain/doc"
      " question, set route='retrieve'.\n3. If conversational, set"
      " route='direct_chat'. If malicious, set route='blocked'."
  )

  result = cast(TriageOutput, structured_triage.invoke([
      SystemMessage(content=system_prompt),
      HumanMessage(content=question),
  ]))

  if not result.is_safe or result.route == "blocked":
    msg = f"Security Blocked: {result.reason}"
    return {
        "route": "blocked",
        "generation": msg,
        "messages": [AIMessage(content=msg)],
    }

  return {
      "route": result.route,
      "file_path": result.extracted_file_path or state.get("file_path"),
  }


def auto_ingestion_node(state: AgentState) -> dict:
  """Automates: Load -> Chunk -> Enrich Metadata -> Hybrid Index."""
  file_path = (state.get("file_path") or "").strip()

  if not file_path or not os.path.exists(file_path):
    msg = f"Ingestion Failed: File '{file_path}' nahi mili."
    return {"generation": msg, "messages": [AIMessage(content=msg)]}

  ext = os.path.splitext(file_path)[-1].lower()

  if ext == ".pdf":
    loader = PyPDFLoader(file_path)
  elif ext in [".docx", ".doc"]:
    loader = Docx2txtLoader(file_path)
  else:
    loader = TextLoader(file_path, encoding="utf-8")

  raw_docs = loader.load()

  splitter = RecursiveCharacterTextSplitter(
      chunk_size=800,
      chunk_overlap=150,
      separators=["\n\n", "\n", ".", " ", ""],
  )
  chunks = splitter.split_documents(raw_docs)

  enriched_docs = []
  base_name = os.path.basename(file_path)
  for idx, c in enumerate(chunks):
    c.metadata["source"] = base_name
    c.metadata["chunk_id"] = f"{base_name}#c{idx+1}"
    c.metadata["page"] = c.metadata.get("page", 1)
    enriched_docs.append(c)

  vector_store.add_documents(enriched_docs)

  msg = (
      f"✅ Ingestion Complete: `{base_name}`\n"
      f"- Sections: {len(raw_docs)}\n"
      f"- Chunks indexed in Qdrant: {len(enriched_docs)}"
  )
  return {"generation": msg, "messages": [AIMessage(content=msg)]}


def direct_chat_node(state: AgentState) -> dict:
  messages = list(state.get("messages", []))
  response = gen_llm.invoke(messages)
  text = extract_text_safely(response.content)
  return {"generation": text, "messages": [AIMessage(content=text)]}


def retrieve_node(state: AgentState) -> dict:
  question = state.get("question", "")
  raw_docs = retriever.invoke(question)

  formatted = []
  for doc in raw_docs:
    formatted.append({
        "content": extract_text_safely(doc.page_content),
        "metadata": {
            "source": doc.metadata.get("source", "Internal Document"),
            "page": doc.metadata.get("page", 1),
            "chunk_id": doc.metadata.get("chunk_id", str(uuid.uuid4())[:6]),
        },
    })
  return {"documents": formatted}


def rerank_node(state: AgentState) -> dict:
  """Advanced RAG: Reranks candidate chunks down to top 3."""
  question = state.get("question", "")
  docs = state.get("documents", [])

  if len(docs) <= 3:
    return {"documents": docs}

  system_prompt = (
      "Given the question and candidate chunks, return ONLY the indices (e.g."
      " 1, 3, 2) of the 3 most relevant chunks separated by comma."
  )
  context_str = "\n".join(
      [f"[{i+1}] {d['content'][:250]}..." for i, d in enumerate(docs)]
  )

  try:
    res = eval_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\n\nChunks:\n{context_str}"),
    ])
    content = res.content if isinstance(res.content, str) else ""
    indices = [
      int(s.strip()) - 1
      for s in content.split(",")
      if s.strip().isdigit()
    ]
    reranked = [docs[i] for i in indices if 0 <= i < len(docs)]
    if not reranked:
      reranked = docs[:3]
  except Exception:
    reranked = docs[:3]

  return {"documents": reranked[:3]}


def grade_documents_node(state: AgentState) -> dict:
  question = state.get("question", "")
  docs = state.get("documents", [])
  combined = "\n\n".join([d["content"] for d in docs])

  grader = eval_llm.with_structured_output(RelevanceScore)
  score = cast(RelevanceScore, grader.invoke([
    SystemMessage(
        content="Assess if context is relevant. Answer 'yes' or 'no'."
    ),
    HumanMessage(content=f"Context:\n{combined}\n\nQuestion:\n{question}"),
  ]))
  return {"documents_relevant": score.binary_score == "yes"}


def rewrite_query_node(state: AgentState) -> dict:
  question = state.get("question", "")
  retry = state.get("retry_count", 0)

  res = eval_llm.invoke([
      SystemMessage(
          content="Rephrase the query for better search. Return only query."
      ),
      HumanMessage(content=question),
  ])
  return {
      "question": extract_text_safely(res.content).strip(),
      "retry_count": retry + 1,
  }


def generate_node(state: AgentState) -> dict:
  question = state.get("question", "")
  docs = state.get("documents", [])

  blocks = [
      f"[{i+1}] (Source: {d['metadata']['source']}, Page"
      f" {d['metadata']['page']})\n{d['content']}"
      for i, d in enumerate(docs)
  ]
  context_str = "\n\n".join(blocks)

  system_prompt = (
      "Answer using ONLY the provided context blocks. Add citations like [1]"
      " after factual claims. If missing, say you do not know."
  )
  res = gen_llm.invoke([
      SystemMessage(content=system_prompt),
      HumanMessage(content=f"Context:\n{context_str}\n\nQuestion:\n{question}"),
  ])
  ans = extract_text_safely(res.content)

  # Footer citations
  sources = {f"- **{d['metadata']['source']}**, Page {d['metadata']['page']}" for d in docs}
  if sources:
    ans += "\n\n### Document Sources\n" + "\n".join(sources)

  return {"generation": ans, "messages": [AIMessage(content=ans)]}


def hallucination_grader_node(state: AgentState) -> dict:
  generation = state.get("generation", "")
  docs = state.get("documents", [])
  combined = "\n\n".join([d["content"] for d in docs])

  grader = eval_llm.with_structured_output(HallucinationScore)
  score = cast(HallucinationScore, grader.invoke([
    SystemMessage(
        content="Verify if generation is grounded in context. 'yes' or 'no'."
    ),
    HumanMessage(
        content=f"Context:\n{combined}\n\nGeneration:\n{generation}"
    ),
  ]))

  if score.binary_score == "no":
    generation += "\n\n*(Warning: Some claims could not be verified in the documents.)*"

  return {"generation": generation}