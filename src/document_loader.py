import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_documents(documents_folder: str) -> list:
    """
    Reads all .txt files from your documents folder.
    Returns a list of Document objects with text + metadata.
    """
    documents = []
    
    # Loop through every file in documents folder
    for filename in os.listdir(documents_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(documents_folder, filename)
            
            # Read the file
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            
            # Skip empty files
            if not text.strip():
                print(f"Skipping empty file: {filename}")
                continue
            
            # Create a Document object
            # metadata tells us WHERE this chunk came from
            doc = Document(
                page_content=text,
                metadata={"source": filename}
            )
            documents.append(doc)
            print(f"Loaded: {filename} ({len(text)} characters)")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def split_documents(documents: list) -> list:
    """
    Splits large documents into smaller chunks.
    Each chunk = ~500 words with 50 word overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # each chunk = ~1000 characters
        chunk_overlap=200,    # 200 character overlap between chunks
                              # overlap ensures we don't cut important info
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    # Test this file
    folder = r"C:\Users\HP\university_assistant\documents"
    docs = load_documents(folder)
    chunks = split_documents(docs)
    
    # Show first chunk as example
    print("\n=== FIRST CHUNK EXAMPLE ===")
    print(f"Source: {chunks[0].metadata['source']}")
    print(f"Content preview: {chunks[0].page_content[:200]}...")