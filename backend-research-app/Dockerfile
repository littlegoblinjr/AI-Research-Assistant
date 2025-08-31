FROM python:3.12.7
WORKDIR /backend-research-app

# OS deps for OCR/PDF; add language packs like tesseract-ocr-hin if you need non-English
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . ./

EXPOSE 8000
# Change backend_research:app if your module/variable names differ
CMD ["uvicorn","backend_research:app","--host","0.0.0.0","--port","8000","--workers","2"]
