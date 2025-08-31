
# AI Research Assistant

**Description:**  
This project is an AI-powered research assistant that allows you to:

- Upload research PDFs or images for analysis  
- Extract text using PDF parsing or OCR  
- Summarize content and answer follow-up questions  
- Store embeddings in Pinecone for efficient retrieval  
- Ask questions based on uploaded documents and indexed knowledge  

The system consists of a **FastAPI backend** and a **React frontend**. LMStudio is used to run local LLMs.

---

## Project Structure

```

project-root/
│
├─ backend-research-app/       # Backend code (FastAPI)
│   ├─ confirmation.py         # Main backend script
│   ├─ requirements.txt
│
├─ frontend/                   # React frontend
│   ├─ package.json
│   ├─ src/
│
├─ docker-compose.yaml         # Docker Compose file to run backend + frontend
└─ .gitignore                  # Git ignore file

````

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed  
- LMStudio running locally (for local LLMs)  
- Pinecone account + API key  

---

## Docker Setup

### Backend Dockerfile

- Installs Python dependencies, PDF/OCR libraries, and your FastAPI app  
- Exposes port `8000`  

### Frontend Dockerfile

- Builds React app  
- Serves it with Nginx on port `8080`  

---

## Run the Project with Docker Compose

From the project root:

```bash
docker compose up --build
````

* Backend will be accessible at: `http://localhost:8000`
* Frontend will be accessible at: `http://localhost:8080`

---

## Example API Call

**Upload a PDF:**

```bash
curl -X POST "http://localhost:8000/ingest/pdf" \
  -F "file=@my_research_paper.pdf"
```

**Ask a question:**

```bash
curl -X POST "http://localhost:8000/chat" \
  -F "message=Explain the key findings of my uploaded PDF."
```

---

## Local LLM (LMStudio) Integration

* Backend uses `OPENAI_BASE` pointing to LMStudio API
* LLMs for keyword extraction and answer generation are configured in the backend

---

## Security Notes

* Pinecone API key should be included securely in the Docker container
* Do **not** commit the API key into the repository

---

## Tips for Windows Users

* Docker Compose setup works on Windows with the backend and frontend ports mapped:

  * Backend: `http://localhost:8000`
  * Frontend: `http://localhost:8080`

* OCR and PDF parsing libraries are included in the backend Dockerfile

---

## Stopping the App

```bash
docker compose down
```

* Stops all running containers
* Keeps images for future use

---

## Future Improvements

* Add support for **read-only Pinecone keys** for safer sharing
* Add **automatic cleanup** of uploaded files after ingestion


```


