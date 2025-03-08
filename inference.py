from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SentenceTransformerEmbeddings

# Define Paths
persist_directory = "/Users/Raneet/Desktop/ChromaQuery/chroma_db"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

llm = ChatGroq(
    temperature=0,
    groq_api_key='your_groq_api_key_here',
    model_name="llama3-8b-8192"
)

chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

def query_chroma(query, db):
    """Retrieve relevant documents and generate an answer using LLM."""
    matching_docs = db.similarity_search(query, k=2)  # Get top-2 relevant docs
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer

query = "Explain Retrieval-Augmented Generation?"
answer = query_chroma(query, new_db)

print("\n📝 **Answer:**")
print(answer)
