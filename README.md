# 📄 DocuChat AI - Production RAG Document Assistant

![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)
![Tailwind](https://img.shields.io/badge/Styling-Tailwind%20CSS-38B2AC?logo=tailwind-css&logoColor=white)
![Python](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![AI](https://img.shields.io/badge/GenAI-Llama--3-blue?logo=meta&logoColor=white)
![Database](https://img.shields.io/badge/Vector%20DB-Pinecone-orange?logo=pinecone&logoColor=white)

**DocuChat AI** is a production-grade Generative AI application that enables users to "chat" with their PDF documents in real-time. Unlike basic wrappers, this project implements a highly optimized **Retrieval Augmented Generation (RAG)** architecture with absolute context isolation, multi-document threading, and built-in hallucination protections.

## 🚀 Key Features

- **🧠 Conversation Memory & Namespaces:** Switch seamlessly between documents using Pinecone vector namespaces. Chat threads remember the context of your last 6 messages while guaranteeing zero "data bleed" when you start a new thread.
- **⚡ Local Vectorization (Free & Fast):** Embeddings are generated mathematically on the CPU using **FastEmbed** (`BAAI/bge-small-en-v1.5`), entirely bypassing costly third-party API rate limits.
- **🔥 Llama-3 Speed:** Inference is running on **Groq's LPUs** (Language Processing Units), achieving token generation speeds roughly 10x faster than traditional GPT-4 APIs structure.
- **📊 Observability & Testing:** Hardened with **LangSmith** for granular tracing of every vector search, alongside a built-in `/evaluate` endpoint powered by **RAGAS** to mathematically destroy AI hallucinations.
- **🛡️ 24/7 Production Availability:** Specifically configured for Render's free tier with a Python `asyncio` heartbeat loop and external UptimeRobot endpoints, entirely preventing server "Cold Starts."
- **🎨 Markdown UI:** A highly polished, monochrome "Vercel-style" UI featuring `react-markdown` to beautifully render AI-generated structured lists and code snippets.

---

## 🛠️ Tech Stack

### **Frontend**
* **Framework:** React.js (Vite)
* **Styling:** Tailwind CSS v4 (Monochrome Dark Theme)
* **Animations:** Framer Motion
* **Rich Text:** React-Markdown
* **Networking:** Axios

### **Backend**
* **API Framework:** FastAPI (Python)
* **LLM Engine:** Llama-3-8B-Instant (via Groq API)
* **Vector Database:** Pinecone (Serverless)
* **Embeddings:** FastEmbed (Local CPU execution)
* **Tracing/Eval:** LangSmith, RAGAS, Datasets
* **PDF Processing:** PyPDF

---

## ⚙️ How It Works (The RAG Pipeline)

1. **Ingestion:** User uploads a PDF. The backend reads it via `PyPDF` and splits it into overlapping 500-character chunks (to preserve sentence context).
2. **Mathematical Vectorization:** Every chunk is passed through local `FastEmbed` and turned into a 384-dimensional mathematical array (vector).
3. **Session Indexing:** The vectors are stored in Pinecone, isolated into a strict `namespace` mapped to your specific UI Session ID.
4. **Retrieval Search:** When you ask a question, the server compares the math of your question to the math of the document chunks using *Cosine Similarity*.
5. **Generation:** The top 5 matching paragraphs from the document + your recent chat history are securely packaged into a system prompt and fed into **Llama-3**, which generates an answer heavily restricted to the document context.

---

## 💻 Installation & Setup

### Prerequisites
* Node.js & npm
* Python 3.11+
* API Keys (Groq, Pinecone, LangSmith[Optional])

### 1. Clone the Repository
```bash
git clone https://github.com/ProgrammerGuy3009/docuchat-rag.git
cd docuchat-rag
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### Configure Environment Variables (`backend/.env`)
```ini
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=docuchat
GROQ_API_KEY=your_groq_key_here

# Optional: Set up LangSmith for tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=docuchat-rag
```

#### Run the Server
```bash
uvicorn main:app --reload
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```
Visit `http://localhost:5173` to launch the application.

---

### 👤 Author
**Prasoon Tripathi**
* LinkedIn: [https://www.linkedin.com/in/programmerguy1001/](https://www.linkedin.com/in/programmerguy1001/)
* GitHub: [https://github.com/ProgrammerGuy3009](https://github.com/ProgrammerGuy3009)
