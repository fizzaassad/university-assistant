import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

print("Testing Groq connection...")

try:
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    response = llm.invoke("Say hello in one sentence.")
    print("Groq works!")
    print("Response:", response.content)
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")