import os
import io
import json
import time
import tempfile
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI  # Changed to AsyncOpenAI

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from pptx import Presentation
from PIL import Image
import pytesseract
from google.cloud import storage
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
import spacy

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise Exception("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'.")

# Initialize FastAPI app
app = FastAPI()

# Initialize DeepSeek client
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",  # DeepSeek API base URL
)

# Performance Metrics Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response: Response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request: {request.url.path} completed in {process_time:.4f} seconds")
    return response

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
def upload_to_firebase(file_bytes: bytes, file_name: str, content_type: str) -> str:
    """Uploads file bytes to Firebase Storage."""
    blob = bucket.blob(f"study_resources/{file_name}")
    blob.upload_from_string(file_bytes, content_type=content_type)
    blob.make_public()
    return blob.public_url

def chunk_text(text: str, max_sentences: int = 5, overlap: int = 1) -> List[str]:
    """Split text into chunks using spaCy."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + max_sentences
        chunk = " ".join(sentences[start:end])
        if chunk:
            chunks.append(chunk)
        start += max(1, max_sentences - overlap)
    return chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from PDF with OCR fallback."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text().strip()
        if len(text) < 20:
            try:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                full_text += ocr_text + "\n"
            except Exception as e:
                logger.warning(f"OCR error on page {page.number}: {e}")
                full_text += "\n"
        else:
            full_text += text + "\n"
    return full_text

def extract_text_from_ppt(ppt_path: str) -> str:
    """Extracts text from PPT/PPTX with OCR."""
    prs = Presentation(ppt_path)
    full_text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                full_text += shape.text + "\n"
            if hasattr(shape, "shape_type") and shape.shape_type == 13:
                try:
                    image_stream = io.BytesIO(shape.image.blob)
                    image = Image.open(image_stream)
                    full_text += pytesseract.image_to_string(image) + "\n"
                except Exception as e:
                    logger.warning(f"Error processing image OCR in PPT: {e}")
    return full_text

def process_file(file_path: str) -> str:
    """Process different file types."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".ppt", ".pptx"]:
        return extract_text_from_ppt(file_path)
    return ""

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using DeepSeek API."""
    try:
        response = await deepseek_client.embeddings.create(
            model="deepseek-embed",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, detail=f"Embedding error: {e}")

async def get_embedding(text: str) -> List[float]:
    """Get single embedding."""
    embeddings = await get_embeddings([text])
    return embeddings[0]

async def index_documents(file_infos: List[Dict]) -> None:
    """Process and index documents with DeepSeek embeddings."""
    vectors = []
    for file_info in file_infos:
        text = process_file(file_info["local_path"])
        if not text.strip():
            continue

        chunks = chunk_text(text)
        if chunks:
            embeddings = await get_embeddings(chunks)
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{os.path.basename(file_info['local_path'])}_{i}"
                metadata = {
                    "source_file": os.path.basename(file_info["local_path"]),
                    "chunk_index": i,
                    "text": chunk,
                    "file_url": file_info["firebase_url"]
                }
                if "user_metadata" in file_info:
                    metadata.update(file_info["user_metadata"])
                vectors.append((vector_id, embedding, metadata))
                await asyncio.sleep(0.2)

    if vectors:
        index.upsert(vectors=vectors)
        logger.info(f"Upserted {len(vectors)} vectors into Pinecone.")

async def deepseek_chat_completion(messages: List[Dict]) -> str:
    """Get chat completion from DeepSeek."""
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(500, detail=f"Chat completion error: {e}")

# -----------------------------------------------------------------------------------
# API Endpoints (Remain Unchanged)
# -----------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
):
    # ... (Same implementation as before)
    # Only changed the internal embedding calls to use DeepSeek client

@app.post("/ask")
async def ask_question(query: QueryRequest):
    # ... (Same implementation as before)
    # Only changed the internal chat completion to use DeepSeek client

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"message": "UniBud API - DeepSeek Edition (OpenAI SDK)"}