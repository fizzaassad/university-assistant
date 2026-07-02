import streamlit as st
import os
import sys
import glob

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

st.set_page_config(
    page_title="University Application Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University & Visa Application Assistant")
st.caption("Powered by real official documents — Germany focus, more countries coming soon")

@st.cache_resource
def load_rag_system():
    # Find documents folder
    root_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(root_dir, "documents")

    # Load all txt files
    documents = []
    for filepath in glob.glob(os.path.join(docs_path, "*.txt")):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(filepath)}
            ))

    if not documents:
        raise ValueError(f"No documents found in {docs_path}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Build vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Build Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt_template = """You are a helpful university and visa application assistant
for Pakistani students applying to study abroad, especially in Germany.

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

    return chain, retriever, docs_path, len(chunks)


# ── Load RAG system ────────────────────────────
try:
    with st.spinner("Loading knowledge base from official documents..."):
        chain, retriever, docs_path, num_chunks = load_rag_system()
    st.success(f"Knowledge base loaded! {num_chunks} chunks from official documents.")
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()


# ── Sample questions ───────────────────────────
st.subheader("Ask a Question")
st.markdown("**Try asking:**")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📄 What documents do I need?"):
        st.session_state.question = "What documents do I need for a German student visa?"
with col2:
    if st.button("💰 How much money do I need?"):
        st.session_state.question = "How much money do I need to study in Germany?"
with col3:
    if st.button("🎓 What scholarships exist?"):
        st.session_state.question = "What scholarships are available for Pakistani students in Germany?"


# ── Question input ─────────────────────────────
question = st.text_input(
    "Your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is the visa fee for Germany?"
)

if question:
    with st.spinner("Searching official documents..."):
        try:
            answer  = chain.invoke(question)
            docs    = retriever.invoke(question)
            sources = list(set([doc.metadata["source"] for doc in docs]))

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            st.info("Answer based on official German Embassy and DAAD documents. Always verify with official sources:")
            st.write("🔗 German Embassy Pakistan: [pakistan.diplo.de](https://pakistan.diplo.de)")
            st.write("🔗 DAAD Scholarships: [daad.de](https://www.daad.de/en/)")
            st.write("🔗 Study in Germany: [study-in-germany.de](https://www.study-in-germany.de)")

        except Exception as e:
            st.error(f"Error getting answer: {e}")


# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.write("""
    This AI assistant answers questions about studying abroad 
    using real official documents from embassies and universities.
    
    It uses RAG (Retrieval-Augmented Generation) to find 
    accurate answers from verified sources — not from AI memory.
    """)

    st.markdown("### Official Sources")
    st.write("🔗 [German Embassy Pakistan](https://pakistan.diplo.de)")
    st.write("🔗 [DAAD Scholarships](https://www.daad.de/en/)")
    st.write("🔗 [Study in Germany](https://www.study-in-germany.de)")
    st.write("🔗 [Make it in Germany](https://www.make-it-in-germany.com)")

    st.markdown("### Countries Covered")
    st.write("🇩🇪 Germany — more countries coming soon")