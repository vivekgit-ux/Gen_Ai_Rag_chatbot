import os
import json
import uuid
import time
from typing import List, Optional
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from indexing.embedding import get_embedding_model


class VectorStoreManager:
    def __init__(
        self,
        collection_name: str = "brics_rag",
        storage_path: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.storage_path = storage_path or os.path.join(base_dir, "qdrant_data")
        self.docstore_path = os.path.join(base_dir, "docstore.json")
        self.embeddings = get_embedding_model()

        if client is not None:
            self.client = client
        else:
            os.makedirs(self.storage_path, exist_ok=True)
            self.client = QdrantClient(path=self.storage_path)

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                test_dim = len(self.embeddings.embed_query("probe"))
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=test_dim, distance=Distance.COSINE),
                )
        except Exception:
            pass

    def save_parent_documents(self, parent_docs: List[Document]):
        store = {}
        if os.path.exists(self.docstore_path):
            try:
                with open(self.docstore_path, "r", encoding="utf-8") as f:
                    store = json.load(f)
            except Exception:
                store = {}

        for doc in parent_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                store[parent_id] = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }

        with open(self.docstore_path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    def get_parent_document(self, parent_id: str) -> Optional[Document]:
        if not os.path.exists(self.docstore_path):
            return None
        try:
            with open(self.docstore_path, "r", encoding="utf-8") as f:
                store = json.load(f)
            data = store.get(parent_id)
            if data:
                return Document(page_content=data["page_content"], metadata=data.get("metadata", {}))
        except Exception:
            return None
        return None

    def _embed_with_retry(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Embeds texts with exponential backoff if Google API rate limits hit."""
        delay = 5.0
        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_documents(texts)
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "resource_exhausted" in err_msg:
                    print(f"[Rate Limit] Hit 429 quota. Pausing {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise e
        return self.embeddings.embed_documents(texts)

    def add_child_chunks(self, child_docs: List[Document], batch_size: int = 80):
        """Indexes child chunks in safe batches with polite delay to respect free-tier RPM."""
        if not child_docs:
            return

        texts = [doc.page_content for doc in child_docs]
        total_chunks = len(child_docs)

        for i in range(0, total_chunks, batch_size):
            batch_docs = child_docs[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]

            # Fetch embeddings with retry protection
            embeddings = self._embed_with_retry(batch_texts)

            points = []
            for doc, emb in zip(batch_docs, embeddings):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                        },
                    )
                )

            if hasattr(self.client, "upsert"):
                self.client.upsert(collection_name=self.collection_name, points=points)
            elif hasattr(self.client, "upload_points"):
                self.client.upload_points(collection_name=self.collection_name, points=points)

            # Polite throttle: 1 second between batches prevents bursting past the 100 RPM ceiling
            if i + batch_size < total_chunks:
                time.sleep(1.0)

    def close(self):
        pass