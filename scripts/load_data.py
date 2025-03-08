import os
from langchain.document_loaders import PyPDFLoader

def load_pdf(pdf_path):
    """Loads a single PDF file and extracts text."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from {pdf_path}")
    return documents

if __name__ == "__main__":
    pdf_path = "/Users/Raneet/Desktop/ChromaQuery/data/2005.11401v4.pdf"  
    docs = load_pdf(pdf_path)
    print(f"🔹 First Page Content:\n{docs[0].page_content[:500]}") 
