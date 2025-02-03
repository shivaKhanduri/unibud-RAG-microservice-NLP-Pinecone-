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
# Load environment variables (configure these in your deployment environment)
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

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# -----------------------------------------------------------------------------------
# Helper Functions (Same as original with minor adjustments)
# -----------------------------------------------------------------------------------
def upload_to_firebase(file_bytes: bytes, file_name: str, content_type: str) -> str:
    """Uploads file bytes to Firebase Storage"""
    blob = bucket.blob(f"study_resources/{file_name}")
    blob.upload_from_string(file_bytes, content_type=content_type)
    blob.make_public()
    return blob.public_url

def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_ppt(ppt_path: str) -> str:
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
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".ppt", ".pptx"]:
        return extract_text_from_ppt(file_path)
    return ""

def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

def index_documents(file_infos: List[Dict]) -> None:
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
            time.sleep(0.2)
    
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
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.filename)[1]) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            file_infos.append({
                "local_path": temp_path,
                "firebase_url": firebase_url
            })
            temp_paths.append(temp_path)
        
        # Index documents
        index_documents(file_infos)
        
        # Cleanup temp files
        for path in temp_paths:
            os.unlink(path)
        
        return {"message": f"Successfully processed {len(files)} files"}
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        # Process query
        query_text = query.query
        if not query_text:
            raise HTTPException(400, detail="Empty query")
        
        # Get embedding and query Pinecone
        query_embedding = get_embedding(query_text)
        response = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        
        # Generate answer with OpenAI
        context = "\n".join([match.metadata["text"] for match in response.matches])
        system_prompt = "You are a knowledgeable teaching assistant. Provide a detailed, step-by-step explanation using the context below."
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
        
        answer = completion.choices[0].message.content
        sources = list({match.metadata["file_url"] for match in response.matches})
        
        return {
            "answer": answer,
            "sources": sources[:2],  # Return top 2 sources
            "token_usage": dict(completion.usage)
        }
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}