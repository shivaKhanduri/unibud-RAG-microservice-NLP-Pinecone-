# unibud-rag-microservice

UniBud API is a FastAPI-based service that enables users to upload study materials (PDFs and PowerPoint files) and then query the content using a chatbot. The service extracts text (using OCR when needed), generates embeddings for text chunks, indexes these embeddings in Pinecone, and finally uses OpenAI's Chat Completion API (GPT-3.5 Turbo) to provide answers to user queries based on the indexed content.

## Features

- **File Upload and Processing:**  
  Supports PDF and PowerPoint (PPT/PPTX) files. If the PDF contains images with text, OCR is applied to extract the text.

- **Text Chunking:**  
  Uses spaCy to split extracted text into manageable chunks with configurable overlap.

- **Embeddings and Indexing:**  
  Generates embeddings for text chunks using OpenAI's `text-embedding-ada-002` model and indexes them in Pinecone for fast similarity search.

- **Chatbot Querying:**  
  Retrieves relevant text chunks from Pinecone based on user queries, builds a context, and uses GPT-3.5 Turbo to generate a detailed answer.

- **File Storage:**  
  Uploads files to Firebase Storage, making them publicly accessible.

- **Usage Monitoring and Caching (Planned):**  
  Can integrate with Firebase authentication to cap the number of questions per user (e.g., 30 queries per month) and cache common answers to reduce API usage.

## Architecture

The following diagram illustrates the overall architecture of the API:

![Untitled diagram-2025-02-02-115117](https://github.com/user-attachments/assets/1754943c-9c5a-4bbb-a7e1-7975480a628d)


> **Diagram Description:**  
> 1. **Frontend/Client:**  
>    - Users interact with the web interface (or any client) to upload files and ask questions.
> 2. **FastAPI Backend:**  
>    - Provides endpoints to upload files (`/upload`) and ask questions (`/ask`).
>    - Middleware for performance monitoring and CORS support.
> 3. **Firebase Storage:**  
>    - Uploaded files are stored in the "study_resources" folder.
> 4. **File Processing Module:**  
>    - Extracts text from PDFs and PowerPoint files using PyMuPDF and python-pptx.
>    - Applies OCR (using Tesseract) when needed.
> 5. **Text Chunking and Embedding Module:**  
>    - Uses spaCy for sentence segmentation and splits text into overlapping chunks.
>    - Generates embeddings for each chunk using OpenAI’s embedding model.
> 6. **Pinecone Vector Database:**  
>    - Stores embeddings for efficient similarity search.
> 7. **Chatbot Querying:**  
>    - When a user asks a question, the query is embedded and used to retrieve relevant chunks from Pinecone.
>    - Constructs a prompt with the retrieved context and forwards it to OpenAI's Chat Completion API to generate the answer.
> 8. **Response:**  
>    - Returns the generated answer along with source information and context details.
