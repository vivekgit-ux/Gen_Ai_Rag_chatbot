import uuid
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParentChildTextSplitter:
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(
        self, documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """
        Splits documents into large parent chunks and smaller child chunks.
        Links child chunks to their parent via parent_id metadata.
        """
        parent_docs = self.parent_splitter.split_documents(documents)
        child_docs = []

        for p_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            p_doc.metadata["parent_id"] = parent_id

            # Create child chunks from the parent chunk
            sub_docs = self.child_splitter.split_documents([p_doc])
            for c_doc in sub_docs:
                c_doc.metadata["parent_id"] = parent_id
                c_doc.metadata["source"] = p_doc.metadata.get("source", "Unknown")
                c_doc.metadata["page"] = p_doc.metadata.get("page", 1)
                child_docs.append(c_doc)

        return parent_docs, child_docs


def split_parent_child(
    documents: List[Document],
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 50,
) -> Tuple[List[Document], List[Document]]:
    """Helper function to split documents into parent and child chunks."""
    splitter = ParentChildTextSplitter(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )
    return splitter.split_documents(documents)