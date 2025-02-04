import os
import io
import json
import time
import tempfile
import logging
from typing import List, Dict, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
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

# Load spaCy model for sentence segmentation (ensure you have installed en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise Exception("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'.")

# Initialize FastAPI app
app = FastAPI()

# Performance Metrics Middleware: measure processing time of requests.
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
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Firebase configuration (read JSON credentials from an environment variable)
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
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket("unibud-22153.appspot.com")

# Pinecone initialization
from pinecone import Pinecone, ServerlessSpec
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
    """Uploads file bytes to Firebase Storage under the 'study_resources' folder."""
    blob = bucket.blob(f"study_resources/{file_name}")
    blob.upload_from_string(file_bytes, content_type=content_type)
    blob.make_public()
    return blob.public_url

def chunk_text(text: str, max_sentences: int = 5, overlap: int = 1) -> List[str]:
    """Split text into chunks of sentences with overlap using spaCy."""
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
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_ppt(ppt_path: str) -> str:
    """Extracts text from a PPT/PPTX file and applies OCR to any images."""
    prs = Presentation(ppt_path)
    full_text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                full_text += shape.text + "\n"
            # Check for picture shapes (pptx image shapes have shape_type value 13)
            if hasattr(shape, "shape_type") and shape.shape_type == 13:
                try:
                    image_stream = io.BytesIO(shape.image.blob)
                    image = Image.open(image_stream)
                    full_text += pytesseract.image_to_string(image) + "\n"
                except Exception as e:
                    logger.warning(f"Error processing image OCR: {e}")
    return full_text

def process_file(file_path: str) -> str:
    """Determines file type by extension and extracts text accordingly."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".ppt", ".pptx"]:
        return extract_text_from_ppt(file_path)
    return ""

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch generate embeddings using OpenAI."""
    try:
        response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, detail=f"Embedding error: {e}")

def get_embedding(text: str) -> List[float]:
    """Generate embedding for a single text by wrapping the batch embedding call."""
    embeddings = get_embeddings([text])
    return embeddings[0]

def index_documents(file_infos: List[Dict]) -> None:
    """Processes provided file infos, splits text into chunks, generates embeddings, and upserts to Pinecone."""
    vectors = []
    for file_info in file_infos:
        text = process_file(file_info["local_path"])
        if not text.strip():
            continue

        chunks = chunk_text(text)
        # Batch process embeddings for all chunks from a file for efficiency
        if chunks:
            embeddings = get_embeddings(chunks)
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{os.path.basename(file_info['local_path'])}_{i}"
                # Merge system generated metadata with user-provided metadata (if any)
                metadata = {
                    "source_file": os.path.basename(file_info["local_path"]),
                    "chunk_index": i,
                    "text": chunk,
                    "file_url": file_info["firebase_url"]
                }
                if "user_metadata" in file_info:
                    metadata.update(file_info["user_metadata"])
                vectors.append((vector_id, embedding, metadata))
                # Sleep a short time to avoid rate limits
                time.sleep(0.2)

    if vectors:
        index.upsert(vectors=vectors)
        logger.info(f"Upserted {len(vectors)} vectors into Pinecone.")

# -----------------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)  # Expecting a JSON string with metadata for each file
):
    start_time = time.time()
    try:
        file_infos = []
        temp_paths = []

        # Parse the metadata JSON if provided
        user_metadata_list = []
        if metadata:
            try:
                user_metadata_list = json.loads(metadata)
                if not isinstance(user_metadata_list, list):
                    raise ValueError("Metadata should be a list of objects, one per file.")
            except Exception as e:
                raise HTTPException(400, detail=f"Invalid metadata JSON: {e}")

        for idx, uploaded_file in enumerate(files):
            # Validate file type
            if not uploaded_file.filename.lower().endswith(('.pdf', '.ppt', '.pptx')):
                raise HTTPException(400, detail="Invalid file type")

            # Read file content
            content = await uploaded_file.read()

            # Upload to Firebase
            firebase_url = upload_to_firebase(
                content,
                uploaded_file.filename,
                uploaded_file.content_type
            )

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.filename)[1]) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            # Prepare the file info; attach user metadata for this file if provided
            file_info = {
                "local_path": temp_path,
                "firebase_url": firebase_url
            }
            if idx < len(user_metadata_list):
                file_info["user_metadata"] = user_metadata_list[idx]
            file_infos.append(file_info)
            temp_paths.append(temp_path)

        # Index documents into Pinecone
        index_documents(file_infos)

        # Cleanup temporary files
        for path in temp_paths:
            os.unlink(path)

        processing_time = time.time() - start_time
        return JSONResponse(status_code=200, content={
            "message": f"Successfully processed {len(files)} files",
            "processing_time": processing_time
        })

    except Exception as e:
        logger.error(f"Error in /upload endpoint: {e}")
        raise HTTPException(500, detail=str(e))

@app.post("/ask")
async def ask_question(query: QueryRequest):
    start_time = time.time()
    try:
        query_text = query.query
        if not query_text.strip():
            raise HTTPException(400, detail="Empty query provided.")

        # Generate an embedding for the query and query Pinecone
        query_embedding = get_embedding(query_text)
        # Increase top_k to get maximum results (here set to 10)
        response = index.query(vector=query_embedding, top_k=10, include_metadata=True)

        # Log the retrieved chunks for debugging
        for match in response.matches:
            logger.info(f"Retrieved chunk: {match.metadata.get('text', '').strip()}")

        # Check if any relevant context was found
        if not response.matches or all(not match.metadata.get("text", "").strip() for match in response.matches):
            processing_time = time.time() - start_time
            return {
                "answer": "I'm sorry, but I couldn't find any relevant information to answer your question.",
                "sources": [],
                "highlighted_context": "",
                "processing_time": processing_time,
                "token_usage": {}
            }

        # Build context from Pinecone matches
        context_chunks = [match.metadata.get("text", "").strip() for match in response.matches if match.metadata.get("text", "").strip()]
        context = "\n".join(context_chunks)
        # Wrap each chunk in <mark> tags for highlighting
        highlighted_context = "\n".join([f"<mark>{chunk}</mark>" for chunk in context_chunks])
        
        system_prompt = (
            "You are a knowledgeable teaching assistant. Use ONLY the context provided below to answer the question. "
            "Provide a detailed, step-by-step explanation. If the context does not contain the answer, explicitly state that no relevant information was found."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query_text}"

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        answer = completion["choices"][0]["message"]["content"]

        sources = []
        for match in response.matches:
            file_url = match.metadata.get("file_url", "No file URL available")
            sources.append(file_url)

        processing_time = time.time() - start_time
        return {
            "answer": answer,
            "sources": sources[:2],
            "context": context,
            "highlighted_context": highlighted_context,
            "processing_time": processing_time,
            "token_usage": dict(completion.get("usage", {}))
        }

    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        raise HTTPException(500, detail=f"An error occurred while processing the query: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the UniBud API! Use /upload to upload files or /ask to query."}
