import os
from typing import List
from PIL import Image
import pytesseract
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)

# Set path for Windows Tesseract OCR binary if present
tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def load_text(file_path: str) -> List[Document]:
    return TextLoader(file_path, encoding="utf-8").load()

def load_pdf(file_path: str) -> List[Document]:
    return PyPDFLoader(file_path).load()

def load_word(file_path: str) -> List[Document]:
    return Docx2txtLoader(file_path).load()

def load_excel(file_path: str) -> List[Document]:
    return UnstructuredExcelLoader(file_path, mode="elements").load()

def load_image_ocr(file_path: str) -> List[Document]:
    text = pytesseract.image_to_string(Image.open(file_path))
    return [
        Document(
            page_content=text,
            metadata={"source": file_path, "type": "image_ocr"}
        )
    ]

def load_document(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext in [".txt", ".md"]:
        return load_text(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return load_word(file_path)
    elif ext in [".xlsx", ".xls"]:
        return load_excel(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return load_image_ocr(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

if __name__ == "__main__":
    docs = load_document("knowledge.txt")
    print(f"Loaded {len(docs)} document(s).")