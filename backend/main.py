"""
DocuChat RAG Backend — V5 Infinite Edition
=============================================
Stack: FastAPI + Pinecone + Groq (Llama-3) + VoyageAI + Gemini + LangSmith + RAGAS
Features: Wave Ingestion, Adaptive RAG, Contextual Chunking, HyDE, Async Parallelism,
          Semantic Cache, Reranker, PII Scrubbing, Gemini Deep Path, SSE Progress
"""

import asyncio
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import re

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pinecone import Pinecone
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
import voyageai
import base64
from google import genai

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
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Render gives us the external URL via this env var (or set it manually)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
KEEP_ALIVE_INTERVAL = int(os.getenv("KEEP_ALIVE_INTERVAL", "600"))  # seconds (10 min)

# LangSmith env vars are read automatically by the SDK:
#   LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

if not PINECONE_API_KEY or not GROQ_API_KEY or not VOYAGE_API_KEY:
    raise ValueError(
        "Missing required API keys. "
        "Ensure PINECONE_API_KEY, VOYAGE_API_KEY, and GROQ_API_KEY are set in your .env file."
    )

if not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY not set — large document contextual chunking will fall back to Groq (truncated).")

# ---------------------------------------------------------------------------
# Global Clients (initialised once at startup)
# ---------------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
voyage_client = voyageai.AsyncClient(api_key=VOYAGE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Server-side state for background ingestion and Gemini Deep Path
ingestion_jobs: dict = {}    # job_id → {status, progress messages, etc.}
document_cache: dict = {}    # session_id → full extracted text (for Gemini Deep Path)

# Backpressure: limit concurrent VoyageAI calls
VOYAGE_SEMAPHORE = asyncio.Semaphore(3)


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
    """Startup / shutdown lifecycle — start keep-alive."""
    logger.info("✅ Clients initialized.")

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
def pii_scrubber(text: str) -> str:
    """Fast Regex PII masking before text hits external APIS or VectorDB."""
    # 1. Credit Cards (13-19 digits)
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED_CC]', text)
    # 2. Emails
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]', text)
    # 3. Phone Numbers
    text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{2,3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', text)
    return text


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


@traceable(name="generate_document_summary")
async def generate_document_summary(full_text: str) -> str:
    """
    Payload-Aware Contextual Chunking:
    - Small docs (≤6000 chars / ~5 pages): Groq 8B summarizes instantly.
    - Large docs (>6000 chars): Gemini Latest handles the full context.
    Returns a 1-2 sentence "DNA" summary to prepend to every chunk.
    """
    summary_prompt = (
        "You are a document summarizer. Read the following document and produce "
        "a single, dense sentence that captures the document's subject, purpose, "
        "and key entities. This sentence will be prepended to every chunk of the "
        "document to provide global context. Output ONLY the summary sentence, nothing else.\n\n"
        f"DOCUMENT:\n{full_text[:50000]}"
    )

    SMALL_DOC_THRESHOLD = 6000  # ~5 pages of text

    if len(full_text) <= SMALL_DOC_THRESHOLD:
        # Small doc → Groq 8B (fast, no external dependency)
        logger.info("📝 Contextual summary via Groq 8B (small doc)")
        resp = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=150,
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    else:
        # Large doc → Gemini Latest (1M token context window)
        if gemini_client:
            logger.info("📝 Contextual summary via Gemini Latest (large doc)")
            resp = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=summary_prompt
            )
            return resp.text.strip()
        else:
            # Fallback: Groq with truncated text if no Gemini key
            logger.warning("📝 Gemini unavailable, falling back to Groq 8B (truncated)")
            truncated_prompt = (
                "You are a document summarizer. Read the following document and produce "
                "a single, dense sentence that captures the document's subject, purpose, "
                "and key entities. Output ONLY the summary sentence.\n\n"
                f"DOCUMENT:\n{full_text[:5500]}"
            )
            resp = await groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": truncated_prompt}],
                max_tokens=150,
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()


@traceable(name="generate_hyde")
async def generate_hyde(user_query: str) -> str:
    """
    HyDE (Hypothetical Document Embeddings):
    Generate a hypothetical answer to improve vector search quality.
    """
    hyde_prompt = (
        "You are a document assistant. Given this user question, write a short "
        "hypothetical paragraph that would answer the question if it existed in a document. "
        "Do NOT say 'I don't know'. Just write the hypothetical content as if it were real.\n\n"
        f"Question: {user_query}"
    )
    resp = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": hyde_prompt}],
        max_tokens=200,
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()


@traceable(name="generate_embeddings")
async def generate_embeddings(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Generate embeddings via VoyageAI Remote API."""
    result = await voyage_client.embed(texts, model="voyage-3", input_type=input_type)
    return result.embeddings


async def embed_with_backpressure(chunks: list[str], input_type: str = "document", batch_size: int = 8) -> list[list[float]]:
    """
    Rate-limited embedding with Semaphore + exponential backoff.
    Processes chunks in micro-batches to avoid 429 errors.
    """
    all_embeddings = []

    async def _embed_batch(batch: list[str]) -> list[list[float]]:
        async with VOYAGE_SEMAPHORE:
            for attempt in range(4):  # max 3 retries
                try:
                    result = await voyage_client.embed(batch, model="voyage-3", input_type=input_type)
                    return result.embeddings
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                        logger.warning(f"⏳ VoyageAI rate-limited, backing off {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        raise
            raise Exception("VoyageAI rate limit exceeded after 4 retries")

    # Fire micro-batches concurrently (limited by semaphore)
    tasks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tasks.append(_embed_batch(batch))

    results = await asyncio.gather(*tasks)
    for batch_result in results:
        all_embeddings.extend(batch_result)

    return all_embeddings


@traceable(name="search_pinecone")
def search_pinecone(query_vector: list[float], top_k: int = 15, namespace: str = "") -> list[str]:
    """Query Pinecone and return list of context document strings."""
    results = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    docs = []
    for match in results.get("matches", []):
        text = match.get("metadata", {}).get("text", "")
        if text:
            docs.append(text)
    return docs

@traceable(name="check_semantic_cache")
def check_semantic_cache(query_vector: list[float]) -> str | None:
    """Check Pinecone cache namespace for identical semantic hits."""
    results = pinecone_index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True,
        namespace="semantic-cache"
    )
    matches = results.get("matches", [])
    if matches and matches[0].get("score", 0) > 0.95:
        return matches[0].get("metadata", {}).get("answer")
    return None

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
        "version": "5.0.0",
        "model": "llama-3.3-70b-versatile",
        "features": ["wave-ingestion", "adaptive-rag", "hyde", "reranker", "semantic-cache", "pii-scrubbing"]
    }


@app.post("/upload-pdf/", tags=["Documents"])
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(..., description="Unique session ID for namespace")
):
    """
    Upload a PDF → return 202 immediately → process in background via waves.
    Subscribe to /ingestion-progress/{job_id} for real-time SSE updates.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    job_id = str(uuid.uuid4())[:8]

    # Initialize job tracking
    ingestion_jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "progress": [],
        "result": None
    }

    # Spawn background task
    background_tasks.add_task(
        _run_ingestion, job_id, contents, file.filename, session_id
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "processing",
            "message": "Ingestion started. Subscribe to /ingestion-progress/{job_id} for live updates.",
        }
    )


async def _run_ingestion(job_id: str, contents: bytes, filename: str, session_id: str):
    """Wave-based background ingestion — processes 25 pages at a time to stay under 512MB RAM."""
    WAVE_SIZE = 25
    start_time = time.time()
    job = ingestion_jobs[job_id]

    def _progress(step: str, detail: str):
        job["progress"].append({"step": step, "detail": detail, "ts": time.time()})
        logger.info(f"📄 [{job_id}] {step}: {detail}")

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        total_pages = len(doc)
        _progress("started", f"Processing {filename} ({total_pages} pages)")

        # ── Phase 1: Extract text from first wave for DNA summary ──
        first_wave_text_parts = []
        for i in range(min(WAVE_SIZE, total_pages)):
            first_wave_text_parts.append(doc[i].get_text())
        first_wave_text = "\n".join(first_wave_text_parts)

        # Vision extraction on first wave
        vision_tasks = []
        for i in range(min(WAVE_SIZE, total_pages)):
            for img in doc[i].get_images(full=True):
                try:
                    base_image = doc.extract_image(img[0])
                    b64_image = base64.b64encode(base_image["image"]).decode('utf-8')
                    vision_tasks.append(
                        groq_client.chat.completions.create(
                            model="llama-3.2-11b-vision-preview",
                            messages=[{"role": "user", "content": [
                                {"type": "text", "text": "Describe this image/chart in extreme detail for database embedding. No intro text."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                            ]}],
                            max_tokens=300
                        )
                    )
                except Exception:
                    pass

        if vision_tasks:
            _progress("vision", f"Analyzing {len(vision_tasks)} images via Vision AI...")
            vision_responses = await asyncio.gather(*vision_tasks, return_exceptions=True)
            for resp in vision_responses:
                if not isinstance(resp, Exception) and resp.choices:
                    first_wave_text += f"\n[IMAGE DATAPOINT]\n{resp.choices[0].message.content}\n"

        # ── Phase 2: Generate DNA summary ──
        _progress("summarizing", "Generating document DNA summary...")
        doc_summary = await generate_document_summary(first_wave_text)
        _progress("summarizing", f"DNA: {doc_summary[:80]}...")

        # ── Phase 3: Wave-based processing ──
        total_chunks_stored = 0
        total_waves = (total_pages + WAVE_SIZE - 1) // WAVE_SIZE
        full_text_accumulator = []  # for Gemini Deep Path cache

        for wave_num in range(total_waves):
            wave_start = wave_num * WAVE_SIZE
            wave_end = min(wave_start + WAVE_SIZE, total_pages)
            _progress("extracting", f"Wave {wave_num + 1}/{total_waves} — pages {wave_start + 1}-{wave_end}")

            # Extract text for this wave
            wave_text_parts = []
            for page_idx in range(wave_start, wave_end):
                page_text = doc[page_idx].get_text()
                wave_text_parts.append(page_text)
            
            wave_text = pii_scrubber("\n".join(wave_text_parts))
            full_text_accumulator.append(wave_text)

            # Chunk this wave's text
            raw_chunks = chunk_text(wave_text)
            if not raw_chunks:
                continue

            # Prepend DNA to every chunk
            chunks = [f"[DOCUMENT CONTEXT]: {doc_summary}\n\n{chunk}" for chunk in raw_chunks]

            # Embed with backpressure
            _progress("embedding", f"Embedding {len(chunks)} chunks (wave {wave_num + 1})...")
            embeddings = await embed_with_backpressure(chunks, input_type="document")

            # Upsert to Pinecone
            _progress("uploading", f"Syncing wave {wave_num + 1} to Pinecone...")
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_vectors = []
                for j in range(i, min(i + batch_size, len(chunks))):
                    global_idx = total_chunks_stored + j
                    batch_vectors.append({
                        "id": f"{filename}-chunk-{global_idx}",
                        "values": embeddings[j],
                        "metadata": {
                            "text": chunks[j],
                            "source": filename,
                            "chunk_index": global_idx,
                        },
                    })
                pinecone_index.upsert(vectors=batch_vectors, namespace=session_id)

            total_chunks_stored += len(chunks)

            # Free memory from this wave
            del wave_text_parts, wave_text, raw_chunks, chunks, embeddings

        # Cache full text for Gemini Deep Path
        document_cache[session_id] = "\n".join(full_text_accumulator)

        elapsed = round(time.time() - start_time, 2)
        _progress("complete", f"Done in {elapsed}s — {total_chunks_stored} chunks stored")

        job["status"] = "complete"
        job["result"] = {
            "filename": filename,
            "pages": total_pages,
            "chunks_stored": total_chunks_stored,
            "processing_time_seconds": elapsed,
        }

        doc.close()
        logger.info(f"✅ [{job_id}] Upload complete: {filename} in {elapsed}s")

    except Exception as e:
        logger.error(f"❌ [{job_id}] Ingestion failed: {e}", exc_info=True)
        job["status"] = "failed"
        job["progress"].append({"step": "error", "detail": str(e), "ts": time.time()})


@app.get("/ingestion-progress/{job_id}", tags=["Documents"])
async def ingestion_progress(job_id: str):
    """
    SSE endpoint — frontend subscribes to get real-time ingestion progress.
    """
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        import json
        last_idx = 0
        while True:
            job = ingestion_jobs.get(job_id)
            if not job:
                break

            # Stream any new progress entries
            progress = job["progress"]
            while last_idx < len(progress):
                entry = progress[last_idx]
                yield f"data: {json.dumps(entry)}\n\n"
                last_idx += 1

            if job["status"] in ("complete", "failed"):
                # Send final result
                if job["result"]:
                    yield f"data: {json.dumps({'step': 'result', 'detail': job['result']})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/", tags=["Chat"])
@traceable(name="chat")
async def chat(request: ChatRequest):
    """V5 Infinite: Router → HyDE + Embed → Cache → Search → Rerank → Adaptive RAG (Fast or Deep Path)"""
    start_time = time.time()

    user_query = pii_scrubber(request.question)

    # ── Phase 1: Router (must complete first to decide the path) ──
    router_prompt = (
        "You are a routing logic system and compliance filter. The user has asked the following message: "
        f"'{user_query}'\n"
        "If this is a simple greeting, a thank you, or a basic conversational statement that DOES NOT require scanning an uploaded document, output EXACTLY the word SIMPLE. "
        "If the user is submitting or asking about a password, credit card, social security number, or explicit personal secret to you, output EXACTLY the word REDACTED. "
        "If this is asking for knowledge, facts, numbers, or requires document analysis, output EXACTLY the word COMPLEX."
    )
    router_resp = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": router_prompt}],
        max_tokens=10,
        temperature=0.0
    )
    router_content = router_resp.choices[0].message.content.lower()

    if "redacted" in router_content:
        async def redacted_stream(): yield "⚠️ System Block: Your query contains sensitive Confidential Information and was redacted before touching our databases."
        return StreamingResponse(redacted_stream(), media_type="text/event-stream")

    if "simple" in router_content:
        completion = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are DocuChat, an AI assistant strictly limited to analyzing uploaded documents. You may respond politely to basic conversational greetings. You MUST refuse to answer general knowledge questions (like 'Who is the president' or 'Tell me about America'). If a user asks a general fact, reply exactly with: 'I am specifically designed to only answer questions regarding the uploaded document.'"},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )
        async def fast_stream():
            async for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta: yield delta
        return StreamingResponse(fast_stream(), media_type="text/event-stream")

    # ── Phase 2: HyDE generation + Single Batch Embedding ──
    # Generate HyDE text first using Groq
    hyde_text = await generate_hyde(user_query)
    
    # Embed BOTH the original query and the HyDE text in a SINGLE Voyage API call.
    # This cuts our embedding requests in half to stay safely under the 3 RPM free limit!
    embed_result = await voyage_client.embed(
        [user_query, hyde_text], 
        model="voyage-3", 
        input_type="query"
    )
    
    query_vector = embed_result.embeddings[0]
    hyde_vector = embed_result.embeddings[1]

    # Blend the original query vector with the HyDE vector (weighted average)
    blended_vector = [
        (q * 0.4 + h * 0.6) for q, h in zip(query_vector, hyde_vector)
    ]

    # ── Phase 3: Cache Check + Pinecone Search (concurrent) ──
    cached_answer, docs = await asyncio.gather(
        asyncio.to_thread(check_semantic_cache, blended_vector),
        asyncio.to_thread(search_pinecone, blended_vector, 15, request.session_id)
    )

    if cached_answer:
        async def stream_cache():
            for word in cached_answer.split(" "):
                yield word + " "
                await asyncio.sleep(0.02)
        return StreamingResponse(stream_cache(), media_type="text/event-stream")

    if not docs:
        async def empty_response(): yield "I don't have any documents indexed yet. Please upload a PDF first."
        return StreamingResponse(empty_response(), media_type="text/event-stream")

    # ── Phase 4: Rerank + Adaptive RAG Decision ──
    rerank_result = await voyage_client.rerank(user_query, docs, model="rerank-2", top_k=5)
    top_score = rerank_result.results[0].relevance_score if rerank_result.results else 0
    final_context = "\n---\n".join([r.document for r in rerank_result.results])

    logger.info(f"🎯 Reranker top score: {top_score:.3f}")

    # ── Phase 5: Adaptive RAG — choose Fast Path or Deep Path ──
    if top_score < 0.5 and gemini_client and request.session_id in document_cache:
        # DEEP PATH: Reranker isn't confident → send full doc to Gemini
        logger.info(f"🧠 Deep Path activated (score {top_score:.3f} < 0.5) — routing to Gemini")
        full_doc_text = document_cache[request.session_id]
        # Truncate to ~900K chars (~450K tokens) to stay within Gemini's context
        truncated = full_doc_text[:900_000]

        gemini_prompt = (
            f"You are DocuChat, a precise document analyst. Answer the user's question based ONLY on the document below.\n\n"
            f"DOCUMENT:\n{truncated}\n\n"
            f"USER QUESTION: {user_query}\n\n"
            f"If the answer is not in the document, say: \"I don't find that information in the uploaded document.\""
        )

        gemini_resp = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=gemini_prompt
        )
        deep_answer = gemini_resp.text

        # Cache this deep answer too
        pinecone_index.upsert(
            vectors=[{
                "id": f"cache-deep-{int(time.time())}",
                "values": blended_vector,
                "metadata": {"answer": deep_answer}
            }],
            namespace="semantic-cache"
        )

        async def stream_deep():
            for word in deep_answer.split(" "):
                yield word + " "
                await asyncio.sleep(0.02)
        return StreamingResponse(stream_deep(), media_type="text/event-stream")

    # FAST PATH: Reranker is confident → use Groq 70B
    system_prompt = (
        "You are DocuChat, a highly precise AI assistant. "
        "You may respond politely to basic conversational greetings. "
        "However, for ANY request for information, facts, or tasks, you MUST answer based ONLY on the provided document Context. "
        "If the user asks for something that is NOT explicitly present in the provided Context or conversation history, you MUST reply: "
        "\"I don't find that information in the uploaded document.\" DO NOT invent answers."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history[-6:]:
        messages.append({"role": "assistant" if msg.role == "bot" else "user", "content": msg.text})
    
    messages.append({"role": "user", "content": f"Context from Document:\n{final_context}\n\nUser Question:\n{user_query}"})

    # Stream to Frontend using massive 70B model
    completion = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stream=True
    )

    async def stream_generator():
        answer_parts = []
        async for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                answer_parts.append(delta)
                yield delta
                
        # Cache the answer for future identical questions
        full_answer = "".join(answer_parts)
        pinecone_index.upsert(
            vectors=[{
                "id": f"cache-{int(time.time())}",
                "values": blended_vector,
                "metadata": {"answer": full_answer}
            }],
            namespace="semantic-cache"
        )
        
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


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
            query_embedding = await generate_embeddings([sample.question], input_type="query")
            query_vector = query_embedding[0]
            docs = await asyncio.to_thread(search_pinecone, query_vector, 5, request.session_id)
            context_text = "\n---\n".join(docs) if docs else ""
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