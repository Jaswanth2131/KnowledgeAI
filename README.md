# 🧠 Multi-Document AI Knowledge Base

![Dashboard Preview](frontend/public/dashboard-preview.png) *(You can add your screenshot here)*

A full-stack, AI-powered knowledge base that allows users to upload, manage, and query multiple documents using Retrieval-Augmented Generation (RAG). Get accurate, context-aware answers across all your documents with precise source citations.

---

## ✨ Features

- **Document Management**: Upload and index multiple file formats (`.pdf`, `.docx`, `.txt`, `.csv`).
- **AI Chat Interface**: ChatGPT-like multi-turn conversational interface with your documents.
- **Semantic Search**: Natural language search with relevance scoring and highlighted matches.
- **Source Citations**: Every AI answer includes exact source citations and page numbers.
- **Fully Local & Private**: Uses **Ollama** for entirely local LLM inference and embeddings. No data is sent to external APIs like OpenAI.
- **Premium Design**: Built with Next.js featuring a stunning dark theme, glassmorphism, and responsive layouts.
- **Analytics Dashboard**: View system performance, query history, and most-referenced documents.

---

## 🏗️ Architecture

- **Frontend**: [Next.js](https://nextjs.org/) (React), Vanilla CSS (Custom Design System), Lucide React Icons.
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) (Python) for robust, async-capable endpoints.
- **RAG Pipeline**: [LangChain](https://python.langchain.com/) for document chunking, orchestration, and LLM chains.
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (persistent local storage).
- **LLM / Embeddings**: [Ollama](https://ollama.com/) (using `llama3` for chat and `nomic-embed-text` for embeddings).

---

## 🚀 Getting Started

Follow these steps to run the project locally on your machine.

### Prerequisites

1. **Python 3.9+** installed.
2. **Node.js 18+** installed.
3. **Ollama** installed. Download from [ollama.com](https://ollama.com/).

### 1. Setup Ollama Models

Before running the backend, pull the necessary open-source models:

```bash
# Language Model for Q&A
ollama pull llama3

# Embedding Model for Vector Search
ollama pull nomic-embed-text
```

### 2. Setup Backend (FastAPI)

Open a terminal and navigate to the backend folder:

```bash
cd backend

# Create a virtual environment (optional but recommended)
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn api:app --reload --port 8000
```
*The backend API will run at `http://localhost:8000`.*

### 3. Setup Frontend (Next.js)

Open a new terminal and navigate to the frontend folder:

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
*The web interface will be available at `http://localhost:3000`.*

---

## 📂 Project Structure

```text
.
├── backend/
│   ├── api.py           # FastAPI server & route handlers
│   ├── config.py        # System configuration (Paths, Chunk Strategy)
│   ├── ingest.py        # Document Parsing, Chunking & ChromaDB insertion
│   ├── rag_engine.py    # Semantic Search, Prompting & RAG generation
│   ├── requirements.txt # Python dependencies
│   └── uploads/         # Local folder for stored files
│
├── frontend/
│   ├── app/
│   │   ├── layout.js    # Sidebar navigation & global layout
│   │   ├── globals.css  # Premium dark theme and design system
│   │   ├── page.js      # Dashboard Homepage
│   │   ├── chat/        # AI Chat Interface
│   │   ├── documents/   # Document Upload & Management
│   │   ├── search/      # Semantic Vector Search
│   │   └── analytics/   # Usage Metrics & History
│   └── package.json     # Node.js dependencies
│
└── README.md
```

---

## 🛠️ Usage

1. **Upload Documents**: Go to the **Documents** page and drop in your `.pdf` or `.txt` files. The system will automatically chunk and embed the text into ChromaDB.
2. **Ask Questions**: Head over to the **AI Chat** page and ask a question. The AI will search your uploaded documents and construct an answer with sources attached.
3. **Search Content**: Use the **Search** page to find exact chunks of text matching your natural language query.
4. **Monitor**: Visit the **Analytics** page to track how many queries have been made, average response times, and what documents are most referenced.

---

## ⚙️ Configuration
You can customize the chunking strategy and models inside `backend/config.py`:
```python
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```
For the frontend, the UI design tokens (colors, sizing, breakpoints) are fully customizable in `frontend/app/globals.css`.

---

## 📝 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute as you see fit!
