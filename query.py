"""
Script to query the ChromaDB vector database using the embedded documents,
and generate answers using a language model.
"""

import sys
import os
import warnings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from config import DB_DIR

# Suppress deprecation warnings for cleaner CLI output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Enable ANSI escape sequences on older Windows terminals
if os.name == 'nt':
    os.system("")

class Colors:
    """ANSI color codes for the terminal."""
    QUESTION = '\033[94m' # Blue
    ANSWER = '\033[92m'   # Green
    SOURCES = '\033[93m'  # Yellow
    ERROR = '\033[91m'    # Red
    RESET = '\033[0m'
    BOLD = '\033[1m'

def get_rag_chain():
    """Builds and returns the Retrieval-Augmented Generation chain."""
    print(f"[{Colors.BOLD}Setup{Colors.RESET}] Initializing Ollama embeddings (model: nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print(f"[{Colors.BOLD}Setup{Colors.RESET}] Loading ChromaDB vector store...")
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 5})

    print(f"[{Colors.BOLD}Setup{Colors.RESET}] Initializing Ollama LLM (model: llama3)...")
    llm = Ollama(model="llama3")

    prompt_template = """You are an expert AI assistant tasked with answering a user's question based strictly on the provided context documents.
Each context document is explicitly labeled with its Source Document identifier.

INSTRUCTIONS:
1. FACTUAL ACCURACY: Analyze the context documents and extract information highly relevant to the user's question.
2. NO HALLUCINATION: Formulate your answer based ONLY on the provided context. If the context does not contain the answer, your exact Answer must be "I don't know". Do not attempt to use outside knowledge.
3. CITATIONS: You MUST include inline citations in your answer referencing the exact Source Document name when stating a fact (e.g., [mock_doc.txt (page 1)]).
4. EVALUATION: Provide your confidence level along with a 1-sentence reasoning for that score based on the explicitness of the context.

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
---
Answer:
<Detailed answer with inline citations>

Confidence:
<High / Medium / Low> - <Reasoning>
---

Context:
{context}

Question:
{input}"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )
    
    document_prompt = PromptTemplate(
        input_variables=["page_content", "filename", "page"],
        template="Source Document: {filename} (page {page})\nContent: {page_content}"
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm, 
        prompt,
        document_prompt=document_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain

def display_sources(docs):
    """Displays the unique source documents used to generate the answer."""
    if not docs:
        print(f"\n{Colors.SOURCES}--- No Source Documents Found ---{Colors.RESET}")
        return
        
    print(f"\n{Colors.BOLD}{Colors.SOURCES}--- Source Documents Retrieved ---{Colors.RESET}{Colors.SOURCES}")
    seen_sources = set()
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown file")
        page = doc.metadata.get("page", "?")
        source_id = f"{filename} (page/section {page})"
        
        if source_id not in seen_sources:
            print(f"- {source_id}")
            seen_sources.add(source_id)
    print(f"----------------------------------{Colors.RESET}\n")

def interactive_loop():
    """Main CLI loop for Q&A with real-time streaming and colors."""
    print(f"\n{Colors.BOLD}--- System Setup ---{Colors.RESET}")
    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        print(f"\n{Colors.ERROR}Could not initialize RAG pipeline: {e}{Colors.RESET}")
        print(f"{Colors.ERROR}Please ensure your embeddings have been populated via ingest.py.{Colors.RESET}")
        sys.exit(1)

    print("\n" + "="*50)
    print(f"{Colors.BOLD}{Colors.QUESTION}=== Production RAG: Multi-Document AI Query ==={Colors.RESET}")
    print("="*50)
    print("Type 'exit' or 'quit' to close the assistant.\n")

    while True:
        try:
            # Colorizes the question prompt
            user_input = input(f"\n{Colors.BOLD}{Colors.QUESTION}Ask a question: {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.SOURCES}Exiting assistant...{Colors.RESET}")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print(f"{Colors.SOURCES}Exiting assistant...{Colors.RESET}")
            break

        print(f"\n{Colors.SOURCES}Thinking...{Colors.RESET}\n")
        
        try:
            # Track context source documents from the first chunk yielded
            source_docs = []
            
            # Start answering color with typing effect
            print(Colors.ANSWER, end="", flush=True)

            # Stream real-time tokens from LCEL chain
            for chunk in rag_chain.stream({"input": user_input}):
                # The first chunk generally gives us the 'context' before the answer starts
                if "context" in chunk and not source_docs:
                    source_docs = chunk["context"]
                
                # Render the LLM's typing effect to the stream
                if "answer" in chunk:
                    sys.stdout.write(chunk["answer"])
                    sys.stdout.flush()
            
            # Terminate the answer color formatting
            print(Colors.RESET)
            
            # Print the separated sources in yellow
            display_sources(source_docs)
            
        except ConnectionError as e:
            print(f"{Colors.ERROR}\nError: Could not connect to local Ollama instance ({e}).{Colors.RESET}")
            print(f"{Colors.ERROR}Please ensure 'ollama serve' is running.{Colors.RESET}")
        except Exception as e:
            err_str = str(e)
            if 'actively refused' in err_str or 'Connection refused' in err_str:
                print(f"{Colors.ERROR}\nError connecting to local Ollama instance. Is 'ollama serve' running?{Colors.RESET}")
            else:
                print(f"{Colors.ERROR}\nAn error occurred during query execution: {e}{Colors.RESET}")

if __name__ == "__main__":
    interactive_loop()
