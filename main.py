import os
import io
import json
import time
import tempfile
from typing import List, Dict
from urllib.parse import quote

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
# Set OpenAI API key and Pinecone API key from environment variables
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
    """Uploads file bytes to Firebase Storage under the 'study_resources' folder."""
    blob = bucket.blob(f"study_resources/{file_name}")
    blob.upload_from_string(file_bytes, content_type=content_type)
    blob.make_public()
    return blob.public_url

def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    """Splits text into chunks of a maximum length."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

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
            if shape.shape_type == 13:  # Picture shape
                try:
                    image_stream = io.BytesIO(shape.image.blob)
                    image = Image.open(image_stream)
                    full_text += pytesseract.image_to_string(image) + "\n"
                except Exception:
                    pass
    return full_text

def process_file(file_path: str) -> str:
    """Determines file type by extension and extracts text accordingly."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".ppt", ".pptx"]:
        return extract_text_from_ppt(file_path)
    return ""

def get_embedding(text: str) -> List[float]:
    """Generates an embedding for the given text using OpenAI's updated API."""
    try:
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {e}")

def index_documents(file_infos: List[Dict]) -> None:
    """Processes provided file infos, splits text into chunks, generates embeddings, and upserts to Pinecone."""
    vectors = []
    for file_info in file_infos:
        text = process_file(file_info["local_path"])
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            vector_id = f"{os.path.basename(file_info['local_path'])}_{i}"
            embedding = get_embedding(chunk)
            metadata = {
                "source_file": os.path.basename(file_info["local_path"]),
                "chunk_index": i,
                "text": chunk,
                "file_url": file_info["firebase_url"]
            }
            vectors.append((vector_id, embedding, metadata))
            time.sleep(0.2)  # To avoid rate limits

    if vectors:
        index.upsert(vectors=vectors)

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

            file_infos.append({
                "local_path": temp_path,
                "firebase_url": firebase_url
            })
            temp_paths.append(temp_path)

        # Index documents into Pinecone
        index_documents(file_infos)

        # Cleanup temporary files
        for path in temp_paths:
            os.unlink(path)

        return {"message": f"Successfully processed {len(files)} files"}

    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        query_text = query.query
        if not query_text.strip():
            raise HTTPException(400, detail="Empty query provided.")

        # Generate an embedding for the query and query Pinecone
        query_embedding = get_embedding(query_text)
        response = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        # Check if any relevant context was found
        if not response.matches or all(not match.metadata.get("text", "").strip() for match in response.matches):
            return {
                "answer": "I'm sorry, but I couldn't find any relevant information to answer your question.",
                "sources": [],
                "token_usage": {}
            }

        # Build context from Pinecone matches
        context = "\n".join([match.metadata.get("text", "") for match in response.matches])
        
        system_prompt = (
            "You are a knowledgeable teaching assistant. Provide a detailed, step-by-step explanation using ONLY the context below. "
            "If the context doesn't contain the answer, explicitly state that no relevant information was found."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query_text}"

        # Create a chat completion using OpenAI API
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

        # Collect sources with fallback if file_url is missing
        sources = []
        for match in response.matches:
            file_url = match.metadata.get("file_url", "No file URL available")
            sources.append(file_url)

        return {
            "answer": answer,
            "sources": sources[:2],  # Return top 2 sources
            "token_usage": dict(completion["usage"])
        }

    except Exception as e:
        raise HTTPException(500, detail=f"An error occurred while processing the query: {e}")



@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the UniBud API! Use /upload to upload files or /ask to query."}
