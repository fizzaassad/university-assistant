from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading vector store...")
vs = Chroma(
    persist_directory=r"C:\Users\HP\university_assistant\chroma_db",
    embedding_function=embeddings
)

print(f"Vector store loaded: {vs._collection.count()} chunks")