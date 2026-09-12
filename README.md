# 🤖 Universal Multimodal RAG Assistant

A production-grade Advanced RAG (Retrieval-Augmented Generation) application built with **Streamlit**, **LangChain**, and the **Google Gemini API**. It supports both general conversation and grounded document analysis across PDFs, spreadsheets, Word documents, and images.

---

## 🚀 Key Features

* **Parent-Child Chunking:** Retains global document context by indexing small chunks for retrieval while feeding larger parent chunks to the LLM.
* **Hybrid Search:** Combines BM25 lexical keyword matching with dense semantic embeddings.
* **Cross-Encoder Reranking:** Filters candidate passages using FlashRank TinyBERT before generation.
* **Inline Grounded Citations:** Traces answers back to specific sources and pages.
* **Multimodal Support:** Processes diagrams, charts, and images natively using Gemini Vision.

---

## 🛠️ Tech Stack

* **LLM & Embeddings:** Google Gemini (`gemini-3.5-flash-lite`, `gemini-embedding-001`)
* **Vector Store:** Qdrant
* **Reranker:** FlashRank (`ms-marco-TinyBERT-L-2-v2`)
* **Frameworks:** LangChain, Streamlit, PyMuPDF, Pandas

---

## 📂 Project Structure

```text
GEN_AI-Application/
├── indexing/
│   ├── document_loader.py    # Multi-format document parser
│   ├── text_splitter.py      # Parent-child chunking logic
│   ├── embedding.py          # Gemini embedding setup
│   └── vector_store.py       # Qdrant client & vector operations
├── retrieval/
│   └── retriever.py          # Hybrid search (BM25 + Dense) & FlashRank reranker
├── generation/
│   └── llm.py                # Grounded response generation & query rewriter
├── evaluation/
│   └── evaluate.py           # RAG benchmarking
├── app.py                    # Streamlit web user interface
├── rag.py                    # Terminal-based testing script
├── .env.example              # Template for environment variables
├── .gitignore                # Prevents uploading secrets & cache
└── requirements.txt          # Python dependencies

## 🚀 Upgrade: Autonomous Agentic RAG Pipeline
This repository has been upgraded from Advanced RAG to a fully autonomous **Agentic RAG** system powered by **LangGraph**:
- **Triage & Intent Routing:** Automatically determines whether to ingest documents, retrieve knowledge, or chat directly.
- **Autonomous Ingestion:** Auto-parses, chunks, embeds (3072-dim Gemini), and updates Qdrant without manual scripts.
- **Self-Correction & Evaluation Loops:** Includes chunk relevance grading, automated query re-writing on failure, and post-generation hallucination verification.
- **Gemini-Inspired Operations UI:** Complete with interactive agent thought-process traces and grounded citations.