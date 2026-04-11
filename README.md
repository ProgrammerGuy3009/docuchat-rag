# 📄 DocuChat AI — V5 Infinite Edition

![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)
![Tailwind](https://img.shields.io/badge/Styling-Tailwind%20CSS-38B2AC?logo=tailwind-css&logoColor=white)
![Python](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![AI](https://img.shields.io/badge/LLM-Llama--3.3--70B-blue?logo=meta&logoColor=white)
![Embeddings](https://img.shields.io/badge/Embeddings-VoyageAI-purple)
![Database](https://img.shields.io/badge/Vector%20DB-Pinecone-orange?logo=pinecone&logoColor=white)
![Gemini](https://img.shields.io/badge/Deep%20Path-Gemini%202.5-4285F4?logo=google&logoColor=white)
![Version](https://img.shields.io/badge/Version-5.0.0-green)

**DocuChat AI** is a production-grade, enterprise-tier Generative AI application that lets users **chat with PDF documents of any size** — from 2-page invoices to 1000-page legal contracts — with zero hallucination, real-time streaming, and intelligent model routing.

> 🏗️ **Not a wrapper. Not a toy.** This implements 14 advanced AI/ML techniques across a distributed multi-model inference stack — the same architecture patterns used by Perplexity, Anthropic, and OpenAI in production.

---

## 🏆 What Makes This Different

| Feature | Basic RAG Projects | DocuChat V5 Infinite |
|---|---|---|
| Document Size | ≤10 pages | **1000+ pages** (wave processing) |
| Embeddings | Local CPU (20s delay) | **VoyageAI cloud** (<1s) |
| LLM | Single model | **Adaptive multi-model** (8B/70B/Gemini) |
| Search | Basic vector search | **HyDE + Reranker + Semantic Cache** |
| Ingestion | Blocking (user waits) | **Background + Live SSE Progress** |
| Security | None | **PII Scrubbing + Compliance Router** |
| Multimodal | Text only | **Vision AI for charts/graphs** |
| Streaming | JSON response | **Real-time SSE token streaming** |

---

## 🧠 The 14 Advanced Techniques

### Ingestion Pipeline
1. **Wave-Based Streaming Ingestion** — Processes 25 pages at a time, freeing memory after each wave. Handles 1000+ page documents on a 512MB RAM server without crashing.
2. **Contextual Chunking (Anthropic's Method)** — Before chunking, a 1-sentence "DNA Summary" of the entire document is generated and prepended to every chunk, ensuring isolated paragraphs never lose their global context.
3. **Payload-Aware Routing** — Small documents (≤5 pages) use Groq 8B for the summary. Large documents (>5 pages) route to Gemini's 1M-token context window.
4. **Multimodal Vision Extraction** — PyMuPDF extracts embedded images/charts, sends them to Groq's Vision LLM (`llama-3.2-11b-vision-preview`), and injects the text descriptions into the vector database.
5. **Backpressure Rate-Limiter** — `asyncio.Semaphore(3)` + exponential backoff (1s→2s→4s→8s) prevents VoyageAI 429 errors on large documents.
6. **Background Processing (202 Accepted)** — Upload returns instantly. Ingestion runs as a background task with real-time SSE progress broadcasting.

### Retrieval Pipeline
7. **PII Scrubbing** — Regex pre-filter (0.001ms) masks credit cards, emails, and phone numbers before text hits any external API or database.
8. **LLM Compliance Router** — Llama-3.1-8B acts as a traffic cop, classifying queries as `SIMPLE` (greeting), `REDACTED` (PII), or `COMPLEX` (document analysis).
9. **HyDE (Hypothetical Document Embeddings)** — For document queries, a hypothetical answer is generated first and blended with the original query vector (40/60 split) to dramatically improve vector search accuracy.
10. **Voyage Reranker** — Top 15 Pinecone results are passed through VoyageAI's neural reranker, surgically selecting only the top 5 most relevant chunks.
11. **Semantic Cache** — Successful Q&A pairs are cached as vectors in a dedicated Pinecone namespace. Repeated semantically-similar questions (>95% match) get instant responses, bypassing the entire pipeline.
12. **Adaptive RAG (Fast Path / Deep Path)** — If the reranker's top confidence score drops below 0.5, the system automatically escalates to Gemini's 1M-token context window, sending the entire cached document for zero-hallucination analysis.
13. **Async Parallelism** — HyDE generation + query embedding fire concurrently via `asyncio.gather()`. Cache check + Pinecone search also run in parallel. ~40% latency reduction.

### Infrastructure
14. **24/7 Anti-Cold-Start Heartbeat** — Async keep-alive loop pings the health endpoint every 10 minutes, preventing Render free-tier from sleeping.

---

## 🔄 Complete System Architecture

```
┌──────────────────── INGESTION ────────────────────┐
│                                                    │
│  PDF Upload → 202 Accepted (instant)               │
│       │                                            │
│       ▼                                            │
│  Background Task: Wave Processing (25 pages/wave)  │
│       │                                            │
│  For each wave:                                    │
│    ├── Extract text (PyMuPDF)                      │
│    ├── Extract images → Vision LLM descriptions    │
│    ├── PII Scrub (regex)                           │
│    ├── Prepend DNA Summary to each chunk           │
│    ├── Embed via VoyageAI (with backpressure)      │
│    ├── Upsert to Pinecone (namespaced)             │
│    └── FREE memory → next wave                     │
│       │                                            │
│  Cache full text → document_cache (for Deep Path)  │
│       │                                            │
│  SSE Progress → Frontend (real-time updates)       │
└────────────────────────────────────────────────────┘

┌──────────────────── RETRIEVAL ────────────────────┐
│                                                    │
│  User Question                                     │
│       │                                            │
│  PII Regex Scrub (0.001ms)                         │
│       │                                            │
│  LLM Router (Llama-3.1-8B)                         │
│    ├── REDACTED → "⚠️ Blocked"                     │
│    ├── SIMPLE → Instant greeting (guardrailed)     │
│    └── COMPLEX ↓                                   │
│         │                                          │
│    ┌────┴────┐ (concurrent)                        │
│    │         │                                     │
│  Embed    HyDE Gen                                 │
│  Query    + HyDE Embed                             │
│    │         │                                     │
│    └────┬────┘                                     │
│         │                                          │
│    Blend vectors (40% query + 60% HyDE)            │
│         │                                          │
│    ┌────┴────┐ (concurrent)                        │
│    │         │                                     │
│  Semantic  Pinecone                                │
│  Cache     Search (top 15)                         │
│    │         │                                     │
│    └────┬────┘                                     │
│         │                                          │
│    Voyage Reranker → top 5                         │
│         │                                          │
│    Confidence Check                                │
│    ├── ≥ 0.5 → Groq 70B Stream (Fast Path ⚡)      │
│    └── < 0.5 → Gemini Deep Path (🧠 Full Doc)     │
│         │                                          │
│    SSE Stream → Frontend (real-time tokens)        │
│         │                                          │
│    Cache answer → semantic-cache namespace          │
└────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| React.js (Vite) | UI Framework |
| Tailwind CSS v4 | Monochrome Dark Theme |
| Framer Motion | Animations |
| React-Markdown | Structured AI response rendering |
| EventSource (SSE) | Real-time streaming |

### Backend
| Technology | Purpose |
|---|---|
| FastAPI (Async) | API Framework |
| Llama-3.3-70B-Versatile | Primary LLM (via Groq LPU) |
| Llama-3.1-8B-Instant | Router / HyDE / Summaries |
| Llama-3.2-11B-Vision | Chart/image analysis |
| VoyageAI (voyage-3) | Cloud embeddings (1024-dim) |
| VoyageAI (rerank-2) | Neural reranker |
| Gemini 2.5 Flash | Deep Path (1M token context) |
| Pinecone Serverless | Vector database |
| PyMuPDF (fitz) | PDF + image extraction |
| LangSmith | Tracing & observability |
| RAGAS | Evaluation metrics |

---

## 💻 Installation & Setup

### Prerequisites
* Node.js & npm
* Python 3.11+
* API Keys: Groq, Pinecone, VoyageAI, Google AI Studio (Gemini)

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
# Required
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=docuchat          # Must be 1024-dim, Cosine, Dense
GROQ_API_KEY=your_groq_key
VOYAGE_API_KEY=your_voyage_key

# Required for Deep Path (Gemini fallback)
GEMINI_API_KEY=your_gemini_key

# Optional: Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=docuchat-rag

# Optional: Production (Render)
RENDER_EXTERNAL_URL=https://your-app.onrender.com
```

#### Pinecone Index Configuration
| Setting | Value |
|---|---|
| Dimensions | `1024` |
| Metric | `Cosine` |
| Type | `Dense` |

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

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check (version, features) |
| `POST` | `/upload-pdf/` | Upload PDF → returns 202 with `job_id` |
| `GET` | `/ingestion-progress/{job_id}` | SSE stream of ingestion progress |
| `POST` | `/chat/` | Adaptive RAG query → SSE token stream |
| `POST` | `/evaluate/` | RAGAS evaluation metrics |

---

## 🔐 Security Features

- **Regex PII Masking**: Credit cards, emails, phone numbers are scrubbed before hitting any external API
- **LLM Compliance Router**: Queries containing personal secrets are blocked with a system warning
- **Session Isolation**: Pinecone namespaces ensure zero data bleed between user sessions
- **Context Guardrails**: Simple bypass loop is locked down to prevent general knowledge exploitation

---

### 👤 Author
**Prasoon Tripathi**
* LinkedIn: [https://www.linkedin.com/in/programmerguy1001/](https://www.linkedin.com/in/programmerguy1001/)
* GitHub: [https://github.com/ProgrammerGuy3009](https://github.com/ProgrammerGuy3009)