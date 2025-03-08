import os
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from load_data import load_pdf
from config import CHROMA_DB_PATH, PDF_PATH

def create_vector_db():
    """Loads PDF, generates embeddings, and stores vectors in ChromaDB."""
    print("🔄 Loading PDF and generating embeddings...")
    docs = load_pdf(PDF_PATH)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_PATH)
    db.persist()

    print(f"✅ ChromaDB saved at: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    create_vector_db()
