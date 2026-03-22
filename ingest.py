"""
Script to load documents from the 'data' directory, process them, generate embeddings,
and store them into the ChromaDB vector database in the 'db' directory.
"""

import os
import hashlib
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import DATA_DIR, DB_DIR

def get_loader(file_path: str) -> BaseLoader:
    """Return the appropriate LangChain loader based on file extension."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        return PyPDFLoader(file_path)
    elif ext == '.txt':
        return TextLoader(file_path, encoding='utf-8')
    elif ext == '.docx':
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def load_documents(data_dir: str) -> List[Document]:
    """Load all supported documents from the data directory."""
    documents = []
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        return documents

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            try:
                loader = get_loader(file_path)
                docs = loader.load()
                
                # Ensure metadata has filename and page number
                for doc in docs:
                    doc.metadata["filename"] = filename
                    # Ensure page is present for consistency
                    # PyPDFLoader usually provides it as 'page'. Other loaders might not.
                    if "page" not in doc.metadata:
                        doc.metadata["page"] = 1
                
                documents.extend(docs)
                print(f"Loaded: {filename} ({len(docs)} pages/sections)")
            except ValueError as e:
                print(f"Skipped {filename}: {e}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def generate_deterministic_ids(chunks: List[Document]) -> List[str]:
    """Generate deterministic IDs for each chunk to ensure idempotency."""
    ids = []
    # keep track of chunk index per filename and page
    source_counters = {}
    
    for chunk in chunks:
        filename = chunk.metadata.get('filename', 'unknown')
        page = chunk.metadata.get('page', 1)
        
        # Create a unique key for the source and page
        key = f"{filename}_p{page}"
        if key not in source_counters:
            source_counters[key] = 0
            
        chunk_index = source_counters[key]
        source_counters[key] += 1
        
        # ID is MD5 hash of filename + page + chunk_index to be safe across runs
        id_str = f"{key}_{chunk_index}"
        chunk_id = hashlib.md5(id_str.encode('utf-8')).hexdigest()
        ids.append(chunk_id)
        
    return ids

def ingest_data():
    """Main ingestion pipeline."""
    print("Starting ingestion process...")
    
    # 1. Load documents from "data" folder
    # 2. Supports PDF, TXT, DOCX
    # 3. Uses LangChain document loaders
    # 7. Adds metadata (filename + page number) applied inside
    documents = load_documents(DATA_DIR)
    if not documents:
        print("No documents to process. Exiting.")
        return

    # 4. Split documents into chunks of size 500 with 100 overlap
    chunks = split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # 5. Use Ollama embeddings with model "nomic-embed-text"
    print("Initializing Ollama embeddings (model: nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 8. Ensure idempotency (generate unique deterministic IDs)
    chunk_ids = generate_deterministic_ids(chunks)

    # 6. Store embeddings in ChromaDB with persistence in a "db" folder
    print("Storing embeddings in ChromaDB...")
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    
    # Add documents to the db (Chroma uses the IDs to update or insert documents uniquely)
    db.add_documents(documents=chunks, ids=chunk_ids)

    # 9. Prints number of chunks created
    print(f"Successfully ingested {len(chunks)} chunks into ChromaDB at '{DB_DIR}'.")

if __name__ == "__main__":
    ingest_data()
