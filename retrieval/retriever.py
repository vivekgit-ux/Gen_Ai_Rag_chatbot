import os
import json
import warnings
import logging
from typing import List, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from indexing.vector_store import VectorStoreManager

warnings.filterwarnings("ignore")
logging.getLogger("flashrank").setLevel(logging.ERROR)


class Retriever:
    def __init__(self, top_k: int = 3, collection_name: str = "brics_rag", client=None):
        self.top_k = top_k
        self.collection_name = collection_name
        self.vector_store = VectorStoreManager(collection_name=self.collection_name, client=client)
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
        self.bm25 = None
        self.bm25_docs: List[Document] = []
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        if os.path.exists(self.vector_store.docstore_path):
            try:
                with open(self.vector_store.docstore_path, "r", encoding="utf-8") as f:
                    store = json.load(f)
                self.bm25_docs = [
                    Document(page_content=v["page_content"], metadata=v.get("metadata", {}))
                    for v in store.values()
                ]
                if self.bm25_docs:
                    corpus_tokens = [
                        d.page_content.lower().replace("?", "").replace(",", "").split()
                        for d in self.bm25_docs
                    ]
                    self.bm25 = BM25Okapi(corpus_tokens)
            except Exception:
                self.bm25_docs = []

    def retrieve(self, query: str) -> List[Document]:
        if not self.bm25:
            self._load_bm25_corpus()

        candidate_parents: dict[str, Document] = {}

        # 1. Dense Semantic Retrieval
        try:
            query_vector = self.vector_store.embeddings.embed_query(query)
            client = self.vector_store.client

            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=self.top_k * 8
                )
                dense_hits = response.points
            elif hasattr(client, "search"):
                dense_hits = client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=self.top_k * 8
                )
            else:
                dense_hits = []

            for hit in dense_hits:
                payload = getattr(hit, "payload", {}) or {}
                p_id = payload.get("metadata", {}).get("parent_id")
                if p_id and p_id not in candidate_parents:
                    parent_doc = self.vector_store.get_parent_document(p_id)
                    if parent_doc:
                        candidate_parents[p_id] = parent_doc
        except Exception:
            pass

        # 2. Sparse BM25 Retrieval
        if self.bm25 and self.bm25_docs:
            clean_tokens = query.lower().replace("?", "").replace(",", "").split()
            scores = self.bm25.get_scores(clean_tokens)
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:self.top_k * 8]

            for idx in ranked_indices:
                if scores[idx] > 0:
                    doc = self.bm25_docs[idx]
                    p_id = doc.metadata.get("parent_id", f"bm25_{idx}")
                    if p_id not in candidate_parents:
                        candidate_parents[p_id] = doc

        candidates = list(candidate_parents.values())
        if not candidates:
            return []

        # 3. FlashRank Cross-Encoder Reranking
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(candidates)]
        req = RerankRequest(query=query, passages=passages)
        ranked = self.ranker.rerank(req)

        results = []
        for r in ranked[:self.top_k]:
            results.append(candidates[r["id"]])

        return results if results else candidates[:self.top_k]

    def format_context(self, docs: List[Document]) -> str:
        formatted_blocks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Document")
            page = doc.metadata.get("page", "N/A")
            formatted_blocks.append(
                f"[Source {i} | Document: {source} | Page: {page}]\n"
                f"{doc.page_content.strip()}"
            )
        return "\n\n" + ("=" * 45) + "\n\n".join(formatted_blocks)