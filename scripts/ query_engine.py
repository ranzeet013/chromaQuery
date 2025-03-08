import os
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SentenceTransformerEmbeddings
from config import CHROMA_DB_PATH, GROQ_API_KEY, MODEL_NAME

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
qa_chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

def query_document(query, top_k=2):
    """Search ChromaDB and use LLM to generate answers."""
    print(f"🔎 Searching for: {query}")
    matching_docs = db.similarity_search_with_score(query, k=top_k)

    if not matching_docs:
        return "No relevant information found."

    answer = qa_chain.run(input_documents=[doc[0] for doc in matching_docs], question=query)
    return answer

if __name__ == "__main__":
    user_query = "Explain Retrieval-Augmented Generation?"
    response = query_document(user_query)
    print(f"🧠 AI Answer:\n{response}")
