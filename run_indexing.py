import os
import sys
import shutil
import pymupdf4llm
from langchain_core.documents import Document

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indexing.text_splitter import ParentChildSplitter
from indexing.vector_store import VectorStoreManager

PDF_FILENAME = "brics_report.pdf"
PDF_PATH = os.path.join(PROJECT_ROOT, PDF_FILENAME)

def main():
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_FILENAME} not found in {PROJECT_ROOT}")
        return

    print(f"1. Parsing {PDF_FILENAME} into Markdown tables...")
    pages_data = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=True)
    raw_docs = []
    for page_num, page in enumerate(pages_data, start=1):
        if isinstance(page, str):
            text = page.strip()
        else:
            text = page.get("text", "").strip()
            page_num = page.get("metadata", {}).get("page", page_num)
        if text:
            raw_docs.append(
                Document(
                    page_content=text,
                    metadata={"source": PDF_FILENAME, "page": page_num}
                )
            )
    print(f"   Extracted {len(raw_docs)} pages.")

    print("2. Splitting into Parent and Child chunks...")
    # Parent chunks keep complete tables intact; child chunks provide vector granularity
    splitter = ParentChildSplitter(
        parent_chunk_size=3000,
        parent_overlap=300,
        child_chunk_size=600,
        child_overlap=100
    )
    parent_docs, child_docs = splitter.split_documents(raw_docs)
    print(f"   Created {len(parent_docs)} parent sections and {len(child_docs)} child chunks.")

    print("3. Storing parents in indexing/docstore.json...")
    vector_mgr = VectorStoreManager(collection_name="brics_rag")
    vector_mgr.save_parent_documents(parent_docs)

    print("4. Embedding child chunks and saving to Qdrant collection 'brics_rag'...")
    vector_mgr.index_child_documents(child_docs)

    print("\n✓ Indexing successfully completed!")

if __name__ == "__main__":
    main()