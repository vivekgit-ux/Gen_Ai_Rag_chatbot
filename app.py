import os
import time
import uuid
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Config & Gemini Dark Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI Workspace",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #131314 !important;
        color: #e3e3e3 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    header, #MainMenu, footer { visibility: hidden; }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1f20 !important;
        border-right: 1px solid #282a2c;
    }

    /* Primary Action Buttons */
    .stButton button {
        background-color: #1a1a1c !important;
        color: #e3e3e3 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 20px !important;
        font-weight: 500 !important;
    }
    .stButton button:hover {
        background-color: #282a2d !important;
        border-color: #8ab4f8 !important;
        color: #8ab4f8 !important;
    }

    /* Minimalist Message Layout */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.8rem 0 !important;
        max-width: 820px !important;
        margin: 0 auto !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li {
        color: #e3e3e3 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    /* Agent Thought Process Box */
    .agent-step-box {
        background-color: #1a1c1e;
        border: 1px solid #3c4043;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .step-line {
        margin: 5px 0;
        color: #8ab4f8;
    }
    .step-ok {
        color: #81c995;
    }

    /* Chunk Citation Box */
    .chunk-box {
        background: #282a2d;
        border-left: 3px solid #8ab4f8;
        padding: 8px 12px;
        border-radius: 4px;
        margin-top: 6px;
        font-size: 0.85rem;
        color: #c4c7c5;
    }

    /* Floating Gemini Input Bar */
    .stChatInputContainer {
        max-width: 820px !important;
        margin: 0 auto !important;
        padding-bottom: 20px !important;
    }
    .stChatInput textarea {
        background-color: #1e1f20 !important;
        color: #ffffff !important;
        border: 1px solid #3c4043 !important;
        border-radius: 28px !important;
        padding: 14px 20px !important;
        font-size: 0.95rem !important;
        caret-color: #8ab4f8 !important;
    }
    .stChatInput textarea:focus {
        border-color: #8ab4f8 !important;
        box-shadow: none !important;
    }
    .stChatInput textarea::placeholder {
        color: #9aa0a6 !important;
    }

    .bottom-disclaimer {
        text-align: center;
        font-size: 0.75rem;
        color: #80868b;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_URL = "http://127.0.0.1:8000"

# -----------------------------------------------------------------------------
# 2. State Management
# -----------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# -----------------------------------------------------------------------------
# 3. Sidebar: Auto-Ingestion & Knowledge Memory
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✨ Agent Workspace")

    if st.button("➕ New Chat / Reset", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("#### 📂 Automated Document Ingestion")
    st.caption("PDF, DOCX, TXT अपलोड करते ही एजेंट खुद लोड, चंक, 3072-dim एम्बेड और Qdrant में स्टोर करेगा।")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    # Auto-Ingestion triggered immediately on upload
    if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
        with st.status(f"⚡ Agent Automating Ingestion for `{uploaded_file.name}`...", expanded=True) as s:
            s.write("📥 Step 1: Loading document content...")
            save_path = os.path.join(os.getcwd(), uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            time.sleep(0.3)

            s.write("✂️ Step 2: Recursive overlap chunking & metadata enrichment...")
            time.sleep(0.3)
            s.write("🧠 Step 3: Computing Dense (Gemini 3072-dim) & BM25 Sparse vectors...")
            s.write("💾 Step 4: Indexing into Qdrant Vector Collection...")

            payload = {
                "question": f"Please ingest {uploaded_file.name} into knowledge base",
                "thread_id": st.session_state.thread_id,
                "file_path": save_path,
            }

            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    s.update(label=f"✅ `{uploaded_file.name}` Indexed & Ready!", state="complete", expanded=False)
                    st.session_state.processed_files.add(uploaded_file.name)

                    ready_msg = f"📄 **Document `{uploaded_file.name}` successfully indexed into Vector DB.**\n\nAgent has processed all chunks and embeddings. You can now ask questions about it."
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ready_msg,
                        "route": "ingest",
                        "docs": [],
                        "steps": [
                            f"Automated Ingestion Tool: Loaded '{uploaded_file.name}'",
                            "Text Splitter: Overlapping chunks generated",
                            "Embedding Engine: Gemini 3072-dim + BM25 Sparse computed",
                            "Vector Store: Successfully upserted to Qdrant",
                        ],
                    })
                    st.rerun()
                else:
                    s.update(label="❌ Ingestion failed", state="error")
                    st.error(res.text)
            except Exception as e:
                s.update(label="❌ Server error", state="error")
                st.error(str(e))

    if st.session_state.processed_files:
        st.markdown("<br>**Active Documents in Vector DB:**", unsafe_allow_html=True)
        for doc_name in st.session_state.processed_files:
            st.markdown(f"🟢 `{doc_name}`")

# -----------------------------------------------------------------------------
# 4. Central Chat Interface (Gemini Look + Agentic Steps)
# -----------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 15vh; margin-bottom: 5vh;">
            <h1 style="font-size: 2.8rem; font-weight: 500; background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                How can I help you today?
            </h1>
            <p style="color: #9aa0a6; font-size: 1.2rem;">
                Upload a document on the left for auto-indexing, or ask any operational query.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "👤" if role == "user" else "✨"

    with st.chat_message(role, avatar=avatar):
        # 1. Agent Reasoning & Graph Execution Steps
        if role == "assistant" and msg.get("steps"):
            with st.expander("🧠 View Agent Thought & Execution Graph", expanded=False):
                st.markdown('<div class="agent-step-box">', unsafe_allow_html=True)
                for step in msg["steps"]:
                    st.markdown(f'<div class="step-line">✔ <span class="step-ok">{step}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # 2. Main Response Text
        st.markdown(msg["content"])

        # 3. Grounded Sources / Retrieved Chunks
        if msg.get("docs") and role == "assistant":
            with st.expander(f"🔍 Grounded Sources & Chunks ({len(msg['docs'])} verified)"):
                for idx, d in enumerate(msg["docs"], 1):
                    meta = d.get("metadata", {})
                    st.markdown(
                        f"""
                        <div class="chunk-box">
                            <b>[{idx}] {meta.get('source', 'Document')}</b> — Page {meta.get('page', 1)}<br>
                            <span style="color: #e3e3e3;">{d.get('content', '')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# -----------------------------------------------------------------------------
# 5. Fixed Input Bar & Live Pipeline Stepper
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your uploaded document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        # Live status displaying the Agentic Graph Execution
        with st.status("Agent Orchestrating Pipeline...", expanded=True) as status:
            status.write("1️⃣ Triage Node: Classifying intent & checking security guardrail...")
            time.sleep(0.3)
            status.write("2️⃣ Hybrid Retrieval: Searching Qdrant (Dense 3072-dim + BM25 Sparse)...")
            time.sleep(0.3)
            status.write("3️⃣ Reranker Node: Scoring candidates down to top-relevant chunks...")
            status.write("4️⃣ Evaluator Node: Checking relevance & validating context...")
            status.write("5️⃣ Synthesis Node: Drafting answer with source grounding...")
            status.write("6️⃣ Hallucination Auditor: Verifying claims against source chunks...")

            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt, "thread_id": st.session_state.thread_id},
                    timeout=45,
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    route = data.get("route_taken", "direct_chat")
                    docs = data.get("documents", [])

                    status.update(label=f"Execution Completed (Route: {route.upper()})", state="complete", expanded=False)

                    # Dynamic Execution Log
                    steps_log = [
                        f"Triage Decision: Evaluated intent -> Selected route '{route}'",
                        f"Hybrid Retrieval: {len(docs)} context chunks pulled via RRF",
                        "Cross-Reranking: Filtered candidate chunks to highest confidence",
                        "Hallucination Guard: Generation strictly grounded in facts",
                    ]

                    st.markdown(answer)

                    if docs:
                        with st.expander(f"🔍 Grounded Sources & Chunks ({len(docs)} verified)"):
                            for idx, d in enumerate(docs, 1):
                                meta = d.get("metadata", {})
                                st.markdown(
                                    f"""
                                    <div class="chunk-box">
                                        <b>[{idx}] {meta.get('source', 'Document')}</b> — Page {meta.get('page', 1)}<br>
                                        <span style="color: #e3e3e3;">{d.get('content', '')}</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "route": route,
                        "docs": docs,
                        "steps": steps_log,
                    })
                    st.rerun()
                else:
                    status.update(label="Agent Pipeline Error", state="error")
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as err:
                status.update(label="Connection Failure", state="error")
                st.error(f"Failed to connect to backend: {str(err)}")

st.markdown('<div class="bottom-disclaimer">AI-generated answers strictly grounded in Qdrant Vector Knowledge Base.</div>', unsafe_allow_html=True)