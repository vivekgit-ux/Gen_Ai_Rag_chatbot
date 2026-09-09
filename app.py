import os
import io
import base64
import tempfile
import warnings
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
import streamlit as st

# Load variables from .env
load_dotenv()

warnings.filterwarnings("ignore")

# Ensure an active API key is available without triggering the duplicate-key warning
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API Key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
    st.stop()

# Set GOOGLE_API_KEY standard and clean duplicate to suppress warnings
os.environ["GOOGLE_API_KEY"] = api_key
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="Universal AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Universal AI Assistant")
st.caption("Ask general prompts or upload any document/image to chat about it.")

# 1,500 requests per day (Free Tier friendly)
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0.3
    )

llm = get_llm()

# Helper to safely convert LLM response to string
def get_response_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([c if isinstance(c, str) else c.get("text", "") for c in content])
    return str(content)

# File extraction
def extract_file_content(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Image
    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        img = Image.open(uploaded_file)
        return {"type": "image", "data": img, "name": uploaded_file.name}
    
    # PDF
    elif ext == ".pdf":
        import pymupdf4llm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        text = pymupdf4llm.to_markdown(tmp_path)
        os.remove(tmp_path)
        return {"type": "text", "data": text, "name": uploaded_file.name}
    
    # CSV / Excel
    elif ext in [".csv", ".xlsx"]:
        df = pd.read_csv(uploaded_file) if ext == ".csv" else pd.read_excel(uploaded_file)
        return {"type": "text", "data": df.to_markdown(index=False), "name": uploaded_file.name}
    
    # Plain text / Word
    else:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        return {"type": "text", "data": text, "name": uploaded_file.name}

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_data" not in st.session_state:
    st.session_state.file_data = None

# Sidebar
with st.sidebar:
    st.header("📎 Upload File (Optional)")
    file = st.file_uploader(
        "Upload Image, PDF, CSV, or Text",
        type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx", "txt", "md"]
    )
    
    if file:
        st.session_state.file_data = extract_file_content(file)
        st.success(f"Loaded: **{file.name}**")
    else:
        st.session_state.file_data = None

    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.session_state.file_data = None
        st.rerun()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask anything, or ask about your uploaded file..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            system_prompt = (
                "You are a helpful, versatile AI assistant.\n"
                "- If a document or image is provided, use its content to answer the user's questions.\n"
                "- If no document is provided, or if the question is unrelated to the document, "
                "answer directly using your general knowledge."
            )

            # Build message payload
            message_content = []

            # 1. Add document text if present
            if st.session_state.file_data and st.session_state.file_data["type"] == "text":
                doc_text = f"--- Attached Document ({st.session_state.file_data['name']}) ---\n"
                doc_text += st.session_state.file_data["data"][:40000]
                message_content.append({"type": "text", "text": doc_text})

            # 2. Add user prompt
            message_content.append({"type": "text", "text": f"\nUser Question: {prompt}"})

            # 3. Add image if present
            if st.session_state.file_data and st.session_state.file_data["type"] == "image":
                buffered = io.BytesIO()
                st.session_state.file_data["data"].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })

            try:
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=message_content)
                ])
                answer = get_response_text(response.content)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")