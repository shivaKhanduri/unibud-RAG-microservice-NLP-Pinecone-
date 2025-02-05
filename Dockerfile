FROM python:3.9-slim

# Install system dependencies (required for PyMuPDF, Tesseract, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Ensure that httpx is installed if it's not already in your requirements.txt
# RUN pip install --no-cache-dir httpx

# Download the spaCy English model
RUN python -m spacy download en_core_web_sm

# Copy the rest of your application code
COPY . .

# Expose the port for Uvicorn
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
