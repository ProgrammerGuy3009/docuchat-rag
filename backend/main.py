"""
DocuChat RAG Backend — Production Build
========================================
Stack: FastAPI + Pinecone + Groq (Llama-3) + FastEmbed + LangSmith + RAGAS
"""

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pinecone import Pinecone
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# LangSmith Tracing — automatically enabled when env vars are set
# ---------------------------------------------------------------------------
try:
    from langsmith import traceable
except ImportError:  # graceful fallback if langsmith isn't installed
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("docuchat")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docuchat")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Render gives us the external URL via this env var (or set it manually)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
KEEP_ALIVE_INTERVAL = int(os.getenv("KEEP_ALIVE_INTERVAL", "600"))  # seconds (10 min)

# LangSmith env vars are read automatically by the SDK:
#   LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError(
        "Missing required API keys. "
        "Ensure PINECONE_API_KEY and GROQ_API_KEY are set in your .env file."
    )

# ---------------------------------------------------------------------------
# Global Clients (initialised once at startup)
# ---------------------------------------------------------------------------
embedding_model = None  # will be set in lifespan
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Keep-Alive Background Task (prevents Render free-tier cold starts)
# ---------------------------------------------------------------------------
async def _keep_alive_loop():
    """
    Pings the app's own health endpoint every 10 minutes so Render
    never considers the service idle and never spins it down.
    """
    if not RENDER_EXTERNAL_URL:
        logger.info("ℹ️  RENDER_EXTERNAL_URL not set — keep-alive ping disabled (local dev).")
        return

    ping_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/"
    logger.info(f"🏓 Keep-alive started — pinging {ping_url} every {KEEP_ALIVE_INTERVAL}s")

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
            try:
                resp = await client.get(ping_url)
                logger.info(f"🏓 Keep-alive ping → {resp.status_code}")
            except Exception as e:
                logger.warning(f"🏓 Keep-alive ping failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle — load the embedding model + start keep-alive."""
    global embedding_model
    logger.info("⏳ Loading FastEmbed model (first run downloads ~90 MB)...")
    from fastembed import TextEmbedding
    embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")  # 384-dim, very fast on CPU
    logger.info("✅ Embedding model ready.")

    # Start the keep-alive background task
    keep_alive_task = asyncio.create_task(_keep_alive_loop())

    yield

    # Cleanup
    keep_alive_task.cancel()
    logger.info("🛑 Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DocuChat RAG API",
    version="2.0.0",
    description="Production RAG backend — upload PDFs and chat with them.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class MessageContext(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[MessageContext] = Field(default_factory=list)
    session_id: str = Field(..., description="Unique session ID to act as a namespace")


class EvalSample(BaseModel):
    question: str
    ground_truth: str  # the expected correct answer


class EvalRequest(BaseModel):
    samples: list[EvalSample] = Field(..., min_length=1, max_length=50)
    session_id: str = Field(..., description="Session ID to evaluate context from")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.
    Smaller chunks (500 chars) give more precise retrieval than the
    original 1000-char chunks, at the cost of slightly more vectors.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


@traceable(name="generate_embeddings")
def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings locally via FastEmbed (CPU, no API calls)."""
    embeddings = list(embedding_model.embed(texts))
    return [e.tolist() for e in embeddings]


@traceable(name="search_pinecone")
def search_pinecone(query_vector: list[float], top_k: int = 5, namespace: str = "") -> str:
    """Query Pinecone and return concatenated context text."""
    results = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    context_parts: list[str] = []
    for match in results.get("matches", []):
        text = match.get("metadata", {}).get("text", "")
        score = match.get("score", 0)
        if text:
            context_parts.append(f"[Relevance: {score:.2f}]\n{text}")
    return "\n---\n".join(context_parts)


@traceable(name="llm_answer")
async def get_llm_answer(context: str, question: str, history: list) -> str:
    """Send the augmented prompt to Groq (Llama-3) and return the answer."""
    system_prompt = (
        "You are DocuChat, a highly precise AI assistant. "
        "You may respond politely to basic conversational greetings (like 'Hi' or 'Hello'). "
        "However, for ANY request for information, facts, or tasks, you MUST answer based ONLY on the provided document Context. "
        "If the user asks for something that is NOT explicitly present in the provided Context or conversation history, you MUST reply: "
        "\"I don't find that information in the uploaded document.\" DO NOT answer general knowledge questions or invent answers. "
        "Cite specific parts of the context when possible."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Inject conversational history (last 6 messages to preserve context)
    for msg in history[-6:]:
        role = "assistant" if msg.role == "bot" else "user"
        messages.append({"role": role, "content": msg.text})

    user_prompt = f"Context from Document:\n{context}\n\nUser Question:\n{question}"
    messages.append({"role": "user", "content": user_prompt})

    completion = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,  # lower = more factual, less creative
        max_tokens=1024,
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
@app.head("/", tags=["Health"])
def health_check():
    """Health check endpoint — used by Render / monitoring services."""
    return {
        "status": "online",
        "service": "DocuChat RAG API",
        "version": "2.0.0",
        "model": "llama-3.1-8b-instant",
    }


@app.post("/upload-pdf/", tags=["Documents"])
@traceable(name="upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(..., description="Unique session ID for namespace")
):
    """
    Upload a PDF → extract text → chunk → embed → store in Pinecone.
    Fully async-safe and handles 300+ page documents efficiently.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    start_time = time.time()
    logger.info(f"📄 Upload started: {file.filename}")

    try:
        # 1. Read and extract text
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        full_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="PDF appears to contain no extractable text.")

        page_count = len(reader.pages)
        logger.info(f"   Pages: {page_count} | Characters: {len(full_text):,}")

        # 2. Chunk the text
        chunks = chunk_text(full_text)
        logger.info(f"   Chunks created: {len(chunks)}")

        # 3. Generate embeddings locally (runs in threadpool to not block event loop)
        embeddings = await asyncio.to_thread(generate_embeddings, chunks)
        logger.info(f"   Embeddings generated: {len(embeddings)}")

        # 4. Upsert to Pinecone in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_vectors = []
            for j in range(i, min(i + batch_size, len(chunks))):
                batch_vectors.append({
                    "id": f"{file.filename}-chunk-{j}",
                    "values": embeddings[j],
                    "metadata": {
                        "text": chunks[j],
                        "source": file.filename,
                        "chunk_index": j,
                    },
                })
            pinecone_index.upsert(vectors=batch_vectors, namespace=session_id)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"✅ Upload complete: {file.filename} in {elapsed}s")

        return {
            "filename": file.filename,
            "status": "Successfully indexed!",
            "pages": page_count,
            "chunks_stored": len(chunks),
            "processing_time_seconds": elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/chat/", tags=["Chat"])
@traceable(name="chat")
async def chat(request: ChatRequest):
    """
    RAG pipeline: embed question → search Pinecone → augment prompt → LLM answer.
    """
    start_time = time.time()

    try:
        user_query = request.question

        # 1. Embed the question
        query_embedding = await asyncio.to_thread(
            generate_embeddings, [user_query]
        )
        query_vector = query_embedding[0]

        # 2. Search Pinecone for relevant context
        context_text = await asyncio.to_thread(
            search_pinecone, query_vector, 5, request.session_id
        )

        if not context_text:
            return {
                "answer": "I don't have any documents indexed yet. Please upload a PDF first.",
                "context_used": "",
                "response_time_seconds": round(time.time() - start_time, 2),
            }

        # 3. Get answer from Groq LLM
        ai_response = await get_llm_answer(context_text, user_query, request.history)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"💬 Chat answered in {elapsed}s")

        return {
            "answer": ai_response,
            "context_used": context_text,
            "response_time_seconds": elapsed,
        }

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/evaluate/", tags=["Evaluation"])
@traceable(name="ragas_evaluate")
async def evaluate(request: EvalRequest):
    """
    RAGAS Evaluation endpoint.
    Send a list of {question, ground_truth} pairs. The system will:
      1. Run each question through the RAG pipeline
      2. Evaluate the answers using RAGAS metrics
      3. Return per-question and aggregate scores
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        questions = []
        ground_truths = []
        answers = []
        contexts = []

        logger.info(f"📊 RAGAS evaluation started with {len(request.samples)} samples")

        # Run each question through the RAG pipeline
        for sample in request.samples:
            query_embedding = await asyncio.to_thread(
                generate_embeddings, [sample.question]
            )
            query_vector = query_embedding[0]
            context_text = await asyncio.to_thread(search_pinecone, query_vector, 5, request.session_id)
            ai_response = await get_llm_answer(context_text, sample.question, [])

            questions.append(sample.question)
            ground_truths.append(sample.ground_truth)
            answers.append(ai_response)
            contexts.append([context_text] if context_text else [""])

        # Build the RAGAS dataset
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Run RAGAS evaluation
        result = await asyncio.to_thread(
            ragas_evaluate,
            eval_dataset,
            metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        )

        logger.info(f"✅ RAGAS evaluation complete")

        return {
            "aggregate_scores": {k: round(v, 4) for k, v in result.items() if isinstance(v, (int, float))},
            "detail": "Evaluation complete. Check LangSmith for full traces.",
        }

    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=f"RAGAS dependencies not installed: {str(e)}. Install with: pip install ragas datasets",
        )
    except Exception as e:
        logger.error(f"❌ Evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------