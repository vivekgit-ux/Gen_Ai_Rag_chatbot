import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()

def get_embedding_model(task_type: str = "RETRIEVAL_DOCUMENT") -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set.")

    os.environ["GOOGLE_API_KEY"] = api_key
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type=task_type
    )

if __name__ == "__main__":
    print("Testing gemini-embedding-001...")
    embedder = get_embedding_model(task_type="RETRIEVAL_DOCUMENT")
    
    sample_text = "Project Alpha drone specifications and flight metrics."
    vector = embedder.embed_query(sample_text)
    
    print("Embedding successful!")
    print(f"Vector dimension: {len(vector)}")
    print(f"Vector preview: {vector[:5]}...")