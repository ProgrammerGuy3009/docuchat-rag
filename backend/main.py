from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pinecone import Pinecone
from groq import Groq
from pydantic import BaseModel
import requests
import io
import os
from dotenv import load_dotenv # NEW IMPORT

# Load the secret .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION (SECURE) ---
# Now it reads from the invisible file instead of your code
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "docuchat"

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if keys loaded correctly (Optional safety check)
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("API Keys not found! Make sure .env file is created.")

# --- SETUP CLIENTS ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    question: str

# --- HELPER FUNCTIONS ---
def get_embeddings_from_cloud(text_chunks):
    """Sends text to HuggingFace Cloud to get embeddings"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text_chunks})
    if response.status_code != 200:
        raise Exception(f"HF API Error: {response.text}")
    return response.json()

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits text with overlap so we don't cut sentences in half"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move forward, but step back a bit (overlap) to catch cut-off sentences
        start += chunk_size - chunk_overlap 
    return chunks

# --- API ENDPOINTS ---
@app.get("/")
def home():
    return {"Status": "AI System Online", "Backend": "FastAPI"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        chunks = get_text_chunks(full_text)
        
        # Process in batches
        vectors = []
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = get_embeddings_from_cloud(batch_chunks)
            
            for j, embedding in enumerate(embeddings):
                vector_id = f"{file.filename}-{i+j}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {"text": batch_chunks[j]}
                })

        index.upsert(vectors=vectors)
        return {"filename": file.filename, "status": "Successfully indexed!", "chunks_stored": len(chunks)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat/")
async def chat(request: ChatRequest):
    """The RAG Logic: Search -> Augmented Prompt -> LLM Answer"""
    try:
        user_query = request.question
        
        # 1. Turn the question into a vector (using the same HF model)
        # We wrap it in a list [user_query] because the API expects a list
        query_embedding = get_embeddings_from_cloud([user_query]) 
        
        # Note: HF returns a list of lists, we just want the first one
        if isinstance(query_embedding, list) and len(query_embedding) > 0:
            query_vector = query_embedding[0]
        else:
            return {"error": "Failed to generate embedding for query"}

        # 2. Search Pinecone for the 3 most relevant chunks
        search_results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

        # 3. Combine the found text into a "Context" block
        context_text = ""
        for match in search_results['matches']:
            context_text += match['metadata']['text'] + "\n---\n"

        # 4. Send to Groq (Llama-3)
        prompt = f"""
        You are a helpful AI assistant. Answer the user's question based ONLY on the context below. 
        If the answer is not in the context, say "I don't find that information in the document."
        
        Context:
        {context_text}
        
        Question: 
        {user_query}
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
        )

        ai_response = chat_completion.choices[0].message.content
        
        return {"answer": ai_response, "context_used": context_text}

    except Exception as e:
        return {"error": str(e)}