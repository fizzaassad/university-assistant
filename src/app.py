import streamlit as st
import os
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import glob

load_dotenv()

# ── Page config ────────────────────────────────
st.set_page_config(
    page_title="University Application Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University & Visa Application Assistant")
st.caption("Powered by real official documents — Germany focus, more countries coming soon")

# ── Load documents ─────────────────────────────
@st.cache_resource
def load_rag_system():
    st.write("Loading knowledge base...")
    
    # Load all txt files from documents folder
    documents = []
    docs_path = os.path.join(os.path.dirname(__file__), '..', 'documents')
    
    for filepath in glob.glob(os.path.join(docs_path, "*.txt")):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if text.strip():
            filename = os.path.basename(filepath)
            documents.append(Document(
                page_content=text,
                metadata={"source": filename}
            ))
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    
    # Build vector store
    embeddings = FakeEmbeddings(size=384)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Build Groq chain
    llm = ChatGroq(
        model="llama3-8b-8192",
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
    
    return chain, retriever

# ── Load system ────────────────────────────────
try:
    chain, retriever = load_rag_system()
    st.success("Knowledge base loaded! Ask me anything about studying in Germany.")
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# ── Chat interface ─────────────────────────────
st.subheader("Ask a Question")

# Sample questions
st.markdown("**Try asking:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What documents do I need?"):
        st.session_state.question = "What documents do I need for a German student visa?"
with col2:
    if st.button("How much money do I need?"):
        st.session_state.question = "How much money do I need to study in Germany?"
with col3:
    if st.button("What scholarships exist?"):
        st.session_state.question = "What scholarships are available for Pakistani students in Germany?"

# Question input
question = st.text_input(
    "Your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is the visa fee for Germany?"
)

if question:
    with st.spinner("Searching official documents..."):
        try:
            # Get answer
            answer = chain.invoke(question)
            
            # Get sources
            docs = retriever.invoke(question)
            sources = list(set([doc.metadata["source"] for doc in docs]))
            
            # Display answer
            st.markdown("### Answer")
            st.write(answer)
            
            # Display sources
            st.markdown("### Sources Used")
            for source in sources:
                st.write(f"📄 {source}")
                
        except Exception as e:
            st.error(f"Error: {e}")

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.write("This assistant answers questions about studying abroad using real official documents.")
    
    st.markdown("### Documents Loaded")
    docs_path = os.path.join(os.path.dirname(__file__), '..', 'documents')
    for f in glob.glob(os.path.join(docs_path, "*.txt")):
        st.write(f"📄 {os.path.basename(f)}")
    
    st.markdown("### Countries Covered")
    st.write("🇩🇪 Germany (more coming soon)")