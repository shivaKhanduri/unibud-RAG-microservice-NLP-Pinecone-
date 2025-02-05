FROM python:3.9-slim

# Install system dependencies (required for PyMuPDF, Tesseract, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optionally ensure that httpx is installed.
# Remove the following line if httpx is already in your requirements.txt.
RUN pip install --no-cache-dir httpx

# Download the spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of your application code
COPY . .

# Expose the port and run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
