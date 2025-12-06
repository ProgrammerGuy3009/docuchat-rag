# 📄 DocuChat AI - Hybrid RAG Document Assistant

![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)
![Tailwind](https://img.shields.io/badge/Styling-Tailwind%20CSS-38B2AC?logo=tailwind-css&logoColor=white)
![Python](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![AI](https://img.shields.io/badge/GenAI-Llama--3-blue?logo=meta&logoColor=white)
![Database](https://img.shields.io/badge/Vector%20DB-Pinecone-orange?logo=pinecone&logoColor=white)

**DocuChat AI** is a full-stack Generative AI application that enables users to "chat" with PDF documents in real-time. Unlike basic wrappers, this project implements a **Hybrid RAG (Retrieval Augmented Generation)** architecture to deliver context-aware answers with sub-200ms latency.

## 🚀 Key Features

-   **🧠 Hybrid AI Architecture:** Runs a lightweight React frontend locally while offloading heavy inference to cloud-based LPU (Language Processing Units) via Groq.
-   **🔍 Semantic Search:** Utilizes **HuggingFace MiniLM** embeddings to understand the *meaning* behind user queries, not just keyword matching.
-   **⚡ High-Speed Inference:** Powered by **Llama-3-8b-Instant**, achieving token generation speeds 10x faster than traditional GPT-4o APIs.
-   **📚 Smart Context Retrieval:** Implemented **Recursive Character Chunking** with overlap to ensure zero data loss across page boundaries.
-   **🎨 Glassmorphism UI:** A modern, responsive "Midnight" interface built with **Tailwind CSS** and **Framer Motion** for smooth animations.

## 🛠️ Tech Stack

### **Frontend**
* **Framework:** React.js (Vite)
* **Styling:** Tailwind CSS (Midnight Theme)
* **Animations:** Framer Motion
* **HTTP Client:** Axios

### **Backend**
* **API Framework:** FastAPI (Python)
* **LLM Engine:** Llama-3 (via Groq API)
* **Vector Database:** Pinecone (Serverless)
* **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
* **PDF Processing:** PyPDF

## ⚙️ How It Works (Architecture)

1.  **Ingestion:** User uploads a PDF → System extracts text and splits it into overlapping chunks (1000 chars).
2.  **Vectorization:** Chunks are converted into 384-dimensional vectors using HuggingFace.
3.  **Indexing:** Vectors are stored in **Pinecone** for high-speed retrieval.
4.  **Retrieval:** When a user asks a question, the system performs a cosine-similarity search to find the top 5 most relevant document sections.
5.  **Generation:** The relevant context + user query are fed into **Llama-3** to generate an accurate, fact-based answer.

## 💻 Installation & Setup

### Prerequisites
* Node.js & npm
* Python 3.8+
* API Keys (Groq, Pinecone, HuggingFace)

### 1. Clone the Repository
* git clone [https://github.com/ProgrammerGuy3009/docuchat-rag.git](https://github.com/ProgrammerGuy3009/docuchat-rag.git)
* cd docuchat-rag

### 2. Backend
* cd backend
* python -m venv venv
**Windows:**
* .\venv\Scripts\activate
**Mac/Linux:**
* source venv/bin/activate
* pip install -r requirements.txt
* **Note**: If requirements.txt is missing, install dependencies manually:
* pip install fastapi uvicorn pypdf pinecone-client groq requests python-multipart python-dotenv
* Configure Environment Variables: Create a .env file in the backend folder:
1. PINECONE_API_KEY=your_pinecone_key
2. HF_TOKEN=your_huggingface_token
3. GROQ_API_KEY=your_groq_key
* Run the Server:
* uvicorn main:app --reload

### 3. Frontend Setup
* Open a new terminal:
* cd frontend
* npm install
* npm run dev
* Visit http://localhost:5173 to launch the application.

### 📸 Screenshots
<img width="1919" height="870" alt="image" src="https://github.com/user-attachments/assets/c3854710-6335-4cd7-bb16-a8689c688d38" />

### 👤 Author
**Prasoon Tripathi**
* https://www.linkedin.com/in/programmerguy1001/
* https://github.com/ProgrammerGuy3009
