import os
import io
import json
import time
import tempfile
from typing import List, Dict, Tuple
from urllib.parse import quote
import spacy
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import fitz  # PyMuPDF
from pptx import Presentation
from PIL import Image
import pytesseract
from google.cloud import storage
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from tenacity import retry, wait_exponential, stop_after_attempt

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------------
# Load NLP models
nlp = spacy.load("en_core_web_sm")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Set OpenAI API key and Pinecone API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Firebase configuration
firebase_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if firebase_creds_json:
    try:
        creds_data = json.loads(firebase_creds_json)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(creds_data, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
    except json.JSONDecodeError:
        raise Exception("Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON")

# Initialize Firebase storage client
storage_client = storage.Client()
bucket = storage_client.bucket("unibud-22153.appspot.com")

# Pinecone initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "unibud-index"
EMBEDDING_DIM = 1536

# Create index if it does not exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# -----------------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------------
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def upload_to_firebase(file_bytes: bytes, file_name: str, content_type: str) -> str:
    """Uploads file bytes to Firebase Storage with retries."""
    blob = bucket.blob(f"study_resources/{file_name}")
    blob.upload_from_string(file_bytes, content_type=content_type)
    blob.make_public()
    return blob.public_url

def chunk_text(text: str, max_sentences: int = 5, overlap: int = 1) -> List[str]:
    """Split text into chunks of sentences with overlap."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + max_sentences
        chunks.append(" ".join(sentences[start:end]))
        start += (max_sentences - overlap)
    return chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_ppt(ppt_path: str) -> str:
    """Extracts text from a PPT/PPTX file with OCR."""
    prs = Presentation(ppt_path)
    full_text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                full_text += shape.text + "\n"
            if shape.shape_type == 13:  # Picture shape
                try:
                    image_stream = io.BytesIO(shape.image.blob)
                    image = Image.open(image_stream)
                    full_text += pytesseract.image_to_string(image) + "\n"
                except Exception:
                    pass
    return full_text

def process_file(file_path: str) -> str:
    """Determines file type and extracts text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".ppt", ".pptx"]:
        return extract_text_from_ppt(file_path)
    return ""

@retry(wait=wait_exponential(multiplier=1, min=1, max=5), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch generate embeddings with exponential backoff."""
    try:
        response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        raise HTTPException(500, detail=f"Embedding error: {e}")

def extract_keywords(text: str) -> List[str]:
    """Extract keywords using OpenAI."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Extract 3-5 key concepts. Return as comma-separated list."
            }, {
                "role": "user",
                "content": text
            }],
            temperature=0.0
        )
        return [kw.strip().lower() for kw in response.choices[0].message.content.split(",")]
    except Exception:
        return []

def index_documents(file_infos: List[Dict]) -> None:
    """Process and index documents with enhanced metadata."""
    batch_size = 32
    vectors = []
    
    for file_info in file_infos:
        text = process_file(file_info["local_path"])
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start+batch_size]
            embeddings = get_embeddings(batch_chunks)
            
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                vector_id = f"{os.path.basename(file_info['local_path'])}_{batch_start+i}"
                metadata = {
                    "source_file": os.path.basename(file_info["local_path"]),
                    "text": chunk,
                    "file_url": file_info["firebase_url"],
                    "keywords": extract_keywords(chunk),
                    "chunk_index": batch_start + i
                }
                vectors.append((vector_id, embedding, metadata))
            time.sleep(0.5)  # Conservative rate limiting

    if vectors:
        # Batch upsert in chunks of 100
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i+100])

# -----------------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        file_infos = []
        temp_paths = []

        for uploaded_file in files:
            if not uploaded_file.filename.lower().endswith(('.pdf', '.ppt', '.pptx')):
                raise HTTPException(400, detail="Invalid file type")

            content = await uploaded_file.read()
            firebase_url = upload_to_firebase(content, uploaded_file.filename, uploaded_file.content_type)

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.filename)[1]) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            file_infos.append({"local_path": temp_path, "firebase_url": firebase_url})
            temp_paths.append(temp_path)

        index_documents(file_infos)

        for path in temp_paths:
            os.unlink(path)

        return {"message": f"Successfully processed {len(files)} files"}

    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        query_text = query.query.strip()
        if not query_text:
            raise HTTPException(400, detail="Empty query provided.")

        # Get query embedding and keywords
        query_embedding = get_embeddings([query_text])[0]
        query_keywords = extract_keywords(query_text)

        # Enhanced Pinecone query with keyword filtering
        response = index.query(
            vector=query_embedding,
            top_k=20,
            filter={"keywords": {"$in": query_keywords}} if query_keywords else {},
            include_metadata=True
        )

        # Rerank with cross-encoder
        matches = response.matches
        if matches:
            pairs = [(query_text, match.metadata["text"]) for match in matches]
            scores = cross_encoder.predict(pairs)
            reranked_matches = [match for _, match in sorted(zip(scores, matches), reverse=True)]
        else:
            reranked_matches = []

        # Prepare context and sources
        context_chunks = [match.metadata["text"] for match in reranked_matches[:5]]
        seen_urls = set()
        sources = []
        
        for match in reranked_matches:
            url = match.metadata.get("file_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(url)
            if len(sources) >= 3:
                break

        # Generate answer with improved prompt
        system_prompt = f"""You are an academic assistant. Rules:
1. Answer using ONLY this context: {" ".join(context_chunks)}
2. If unsure, state "I couldn't find sufficient information."
3. Cite sources like [1], [2] from: {", ".join(sources[:3]) or 'No sources'}"""
        
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_text}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        return {
            "answer": completion.choices[0].message.content,
            "sources": sources[:3],
            "token_usage": completion.usage.dict()
        }

    except Exception as e:
        raise HTTPException(500, detail=f"Query processing error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"message": "Enhanced UniBud API with intelligent retrieval"}