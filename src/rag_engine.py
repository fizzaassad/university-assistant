import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from document_loader import load_documents, split_documents

load_dotenv()

DOCUMENTS_FOLDER = r"C:\Users\HP\university_assistant\documents"
CHROMA_DB_PATH   = r"C:\Users\HP\university_assistant\chroma_db"


from langchain_community.embeddings import FakeEmbeddings

def get_embeddings():
    # Simple embeddings that work without downloading any model
    return FakeEmbeddings(size=384)

def build_vector_store(chunks):
    print("Building vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DB_PATH
    )
    print(f"Vector store built with {len(chunks)} chunks!")
    return vector_store


def load_vector_store():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings()
    )


def build_rag_chain(vector_store):
    print("Connecting to Groq AI...")

    # Free Groq model — Llama 3 is powerful and completely free
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt_template = """You are a helpful university and visa application assistant 
for students applying to study abroad, especially in Germany.

Use ONLY the following context from official documents to answer the question.
If the answer is not in the context, say "I don't have that specific information 
in my documents. Please check the official embassy or university website."

Always mention which document your answer came from.

Context from official documents:
{context}

Student's Question: {question}

Helpful Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(chain, retriever, question: str) -> dict:
    answer  = chain.invoke(question)
    docs    = retriever.invoke(question)
    sources = list(set([doc.metadata["source"] for doc in docs]))

    return {
        "answer":  answer,
        "sources": sources
    }

if __name__ == "__main__":
    if not os.path.exists(CHROMA_DB_PATH):
        print("Building vector store for first time...")
        docs   = load_documents(DOCUMENTS_FOLDER)
        chunks = split_documents(docs)
        vs     = build_vector_store(chunks)
    else:
        print("Loading existing vector store...")
        vs = load_vector_store()

    chain, retriever = build_rag_chain(vs)

    print("\nTesting RAG system...")
    
    try:
        result = ask_question(
            chain, retriever,
            "What documents do I need for a German student visa?"
        )
        print("\n=== ANSWER ===")
        print(result["answer"])
        print("\n=== SOURCES USED ===")
        for source in result["sources"]:
            print(f"  - {source}")
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")

   