import datetime
from uuid import uuid4
import os
import io
import ast
import re
import json
import urllib.request
import urllib.parse
from typing import List, Optional
from starlette.concurrency import run_in_threadpool
from fastapi import Request, Form, UploadFile, File, BackgroundTasks
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pytesseract
from dotenv import load_dotenv

# Optional heavy deps – we guard usage so app still starts if some are missing
import PyPDF2
from pdf2image import convert_from_bytes
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()  # defaults to searching for .env in the current working directory

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ============ CONFIG / ENV ============
OPENAI_BASE = os.getenv("OPENAI_BASE", "http://127.0.0.1:1234/v1")  # LM Studio default
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lmstudio")
OPENAI_MODEL_KEYWORD = os.getenv("OPENAI_MODEL_KEYWORD", "gemma-3-4b-it")
OPENAI_MODEL_ANSWER = os.getenv("OPENAI_MODEL_ANSWER", "gemma-3-4b-it")

#PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "research-paper-storage")
PINECONE_CHAT_INDEX = os.getenv("PINECOONE_CHAT_INDEX", "chat-history")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")

ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.4"))


# ============ FASTAPI APP ============
app = FastAPI(title="Research Assistant API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ CLIENTS ============
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX)
index_chat = pc.Index(PINECONE_CHAT_INDEX)

llm_keywords = ChatOpenAI(
    base_url=OPENAI_BASE, model=OPENAI_MODEL_KEYWORD, temperature=0, api_key=OPENAI_API_KEY
)
llm_answer = ChatOpenAI(
    base_url=OPENAI_BASE, model=OPENAI_MODEL_ANSWER, temperature=0, api_key=OPENAI_API_KEY
)

few_shot_search_prompt = ChatPromptTemplate.from_template(
    """
You are an expert at creating search queries for a research document database. 
Given a user's question, your task is to rewrite it in a way that maximizes the chance
of retrieving the most relevant documents.

Examples:

User Query: "Explain convolutional neural networks"
Search Query: "cnn convolutional-layer pooling-layer deep-learning"

User Query: "What is transfer learning?"
Search Query: "transfer-learning pretrained-model fine-tuning domain-adaptation"

Now rewrite the following user query into a search-friendly query:

User Query: {query}
Search Query:
"""
)

search_chain = few_shot_search_prompt | llm_keywords | StrOutputParser()


keyword_prompt = ChatPromptTemplate.from_template(
    """You are an expert in identifying the **main subject or technology keywords** from a query. Your task is to extract only the most relevant and significant keywords related to the core subject or technology mentioned in the query, ignoring less relevant terms or any fluff.

Your output should be **only the most important keywords** in the format shown below. Be sure to focus on **primary concepts** and **technical terms**. Do not include general or ambiguous terms.

Example format:
['GenAI', 'AI']
['Transformers', 'Bert']

**Instructions**:
1. Focus on **main subjects** and **technology keywords** related to the core topic of the query.
2. Ignore general terms or common words that don't represent the main focus of the query.
3. Return only the keywords that clearly represent the technical concepts or main subjects discussed.
4. Ensure its entirely in lowercase and if you are giving a more than one word keyword, instead of space use dash like '-'

**Query**: {query}
"""
)
keyword_chain = keyword_prompt | llm_keywords | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_template(
    """Answer the query based on the provided context. If old conversations are present and are relevant to the query, consider them in your answer. Ensure the answer includes a link to the paper that supports it.

**Query**: {question}

**Context**: {context}

Provide the **source link** for the answer only if available and please dont leave out other important details if you dont find a source link.

Make sure to consider equations if you come across any.
"""
)


# ============ SCHEMAS ============
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gpt-research-pro"  # forwarded but not required by backend
    creativity: Optional[int] = 30  # 0..100 maps to 0..1 temp


class ChatResponse(BaseModel):
    reply: str


class IngestResponse(BaseModel):
    chunks: int


# ============ HELPERS ============
def creativity_to_temp(v: Optional[int]) -> float:
    try:
        n = int(v or 0)
        n = max(0, min(100, n))
        return round(n / 100.0, 2)
    except Exception as e:
        logging.error(f"Error converting creativity: {e}")
        return 0.0


def extract_keywords(query: str) -> List[str]:
    # prefer LLM keywords; fallback to a simple regex split
    if keyword_chain is not None:
        try:
            out = keyword_chain.invoke({"query": query})
            return ast.literal_eval(out)
        except Exception as e:
            logging.error(f"Error extracting keywords from query: {query}. Error: {e}")
            pass
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", query)][:5]


def pinecone_search_chat(q, top_k: int = 1):
    if index_chat is None:
        logging.warning("Pinecone chat index is None.")
        return {'results': {'hits': []}}

    logging.info(f"Searching Pinecone chat index with query: {q}")
    return index_chat.search(
        namespace="__default__",
        query={'inputs': {'text': q}, 'top_k': top_k},
        fields=["text"]
    )


def pinecone_search_text(q: str, top_k: int = 4):
    if index is None:
        logging.warning("Pinecone index is None.")
        return {"result": {"hits": []}}

    logging.info(f"Searching Pinecone text index with query: {q}")
    return index.search(
        namespace="__default__",
        query={"inputs": {"text": q}, "top_k": top_k},
        fields=["text", "link", "summary"],
    )


def build_context_from_hits(hits) -> str:
    ctx = ""
    for h in (hits or {}).get("result", {}).get("hits", []):
        f = h.get("fields", {}) or {}
        text = f.get("text")

        link = f.get("link")
        if text:
            ctx += f"Text: {text}\n"
            if link:
                ctx += f"Link: {link}\n"
            ctx += "\n"
    return ctx.strip()


def answer_with_context(question: str, user_id, context: str, temperature: float, mode) -> str:
    logging.info(f"Answering with context for question: {question}")
    if mode == "direct":
        chain_final = answer_prompt | llm_answer.bind(temperature=temperature) | StrOutputParser()
        final_res = chain_final.invoke({"question": question, "context": context})
        mongo_prompt = answer_prompt.format(question=question, context=context)

        # Pinecone upsert
        record = {
            "_id": f"chat-{uuid4()}",
            "text": f"Q: {question}\n\n--- AI ---\n{final_res}",
            "type": "chat",
        }

        logging.info("Upserting chat record into Pinecone.")
        index_chat.upsert_records(namespace="__default__", records=[record])
        return getattr(final_res, "content", str(final_res))

    # Step 1: Split context into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    docs = [Document(page_content=context)]
    splits = splitter.split_documents(docs)
    chunks = [sp.page_content for sp in splits]

    # Step 2: Summarize each chunk individually
    partial_summaries = []
    for chunk in chunks:
        chain = answer_prompt | llm_answer.bind(temperature=temperature) | StrOutputParser()
        res = chain.invoke({"question": question, "context": chunk})
        partial_summaries.append(getattr(res, "content", str(res)))

    # Step 3: Combine partial summaries into a final answer
    combined_context = "\n\n".join(partial_summaries)
    chain_final = answer_prompt | llm_answer.bind(temperature=temperature) | StrOutputParser()
    final_res = chain_final.invoke({"question": question, "context": combined_context})

    formatted_prompt = answer_prompt.format(question=question, context=combined_context)

    # Step 4: Store in Pinecone as usual
    mongo_prompt = str(formatted_prompt)
    logging.info(f"Upserting final answer into Pinecone.")
    record = {
        "_id": f"chat-{uuid4()}",
        "text": f"Q: {question}\n\n--- AI ---\n{final_res}",
    }
    index_chat.upsert_records(namespace="__default__", records=[record])

    return getattr(final_res, "content", str(final_res))


def read_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        logging.info(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return ""


def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes)
        text = ""
        for img in pages:
            text += pytesseract.image_to_string(img)
        logging.info(f"OCR PDF text length: {len(text)} characters.")
        return text
    except Exception as e:
        logging.error(f"Error during OCR on PDF: {e}")
        return ""


def ocr_image_bytes(img_bytes: bytes) -> str:
    try:
        text = pytesseract.image_to_string(io.BytesIO(img_bytes))
        logging.info(f"OCR Image text length: {len(text)} characters.")
        return text
    except Exception as e:
        logging.error(f"Error during OCR on image: {e}")
        return ""


def upsert_text_chunks(full_text: str, source: str, link: Optional[str] = None, summary: Optional[str] = None) -> int:
    if index is None or not full_text.strip():
        logging.warning("No text to upsert into Pinecone.")
        return 0

    run_id = str(uuid4())
    splitter = RecursiveCharacterTextSplitter(chunk_size=14000, chunk_overlap=500)
    docs = [Document(page_content=full_text)]
    splits = splitter.split_documents(docs)
    chunks = [sp.page_content for sp in splits]

    records = []
    for i, chunk_text in enumerate(chunks):
        rec_id = f"{source}:{run_id}:{i}"

        records.append({
            "_id": rec_id,
            "text": f"--DOC_CONTENT--\n{chunk_text}--SUMMARY--\n{summary}",
            "source": source,
            "run_id": run_id,
            "chunk_number": i,
            "link": link,
        })

    if records:
        logging.info(f"Upserting {len(records)} records into Pinecone.")
        index.upsert_records(namespace="__default__", records=records)
    return len(records)


def arxiv_ingest_for_keywords(keywords: List[str], user_query: str):
    for kw in keywords:
        try:
            # URL-encode the keyword
            encoded_kw = urllib.parse.quote(kw)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_kw}&start=0&max_results={ARXIV_MAX_RESULTS}"
            data = urllib.request.urlopen(url)
            xml = data.read().decode("utf-8")
            soup = BeautifulSoup(xml, "xml")
            for entry in soup.find_all("entry"):
                pdf_link_tag = entry.find("link", {"type": "application/pdf"})
                summary_tag = entry.find("summary")
                summary = (summary_tag.text or "").strip() if summary_tag else None
                if not pdf_link_tag:
                    continue
                pdf_link = pdf_link_tag.get("href")

                # Fetch PDF
                resp = requests.get(pdf_link)
                text = read_pdf_bytes(resp.content)
                if not text:
                    text = ocr_pdf_bytes(resp.content)
                    if not text:
                        continue

                upsert_text_chunks(text, source="arxiv-paper", link=pdf_link, summary=summary)
        except Exception as e:
            logging.error(f"Error during arXiv ingestion for keyword '{kw}': {e}")
            continue
# ============ ENDPOINTS ============
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    background: BackgroundTasks,
    message: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    creativity: Optional[str] = Form(None),
    augment: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    logging.info("Processing chat request...")
    ct = request.headers.get("content-type", "")
    if "application/json" in ct:
        data = await request.json()
        body_message = data.get("message")
        body_model = data.get("model")
        body_creativity = data.get("creativity")
        user_id = data.get("user_id")
        body_augment = (data.get("augment") or "file_plus_index").lower()
        uploaded_file = None
    else:
        body_message = message
        body_model = model
        try:
            body_creativity = int(creativity) if creativity is not None else None
        except Exception as e:
            logging.error(f"Error parsing creativity: {e}")
            body_creativity = None
        body_augment = (augment or "file_plus_index").lower()
        uploaded_file = file

    if not body_message and not uploaded_file:
        return ChatResponse(reply="Please provide a message or attach a PDF.")

    temp = creativity_to_temp(body_creativity)

    """previous_chats = pinecone_search_chat(body_message, top_k=1)
    previous_context = ""
    hits = (previous_chats or {}).get("result", {}).get("hits", [])
    for h in hits:
        f = h.get("fields", {}) or {}
        previous_context += f.get("text", "") + "\n\n"""
        
    SCORE_THRESHOLD_CHAT = 0.3

    previous_chats = pinecone_search_chat(body_message, top_k=1)
    previous_context = ""
    hits = (previous_chats or {}).get("result", {}).get("hits", [])

    # Loop through the hits and apply the threshold condition
    for h in hits:
        score = float(h.get("_score", 0.0))  # Get the score of the hit
        if score >= SCORE_THRESHOLD_CHAT:
            f = h.get("fields", {}) or {}
            previous_context += f.get("text", "") + "\n\n"
        else:
            logging.info(f"Skipping hit with score below threshold: {score}")

    file_context = ""
    if uploaded_file is not None:
        b = await uploaded_file.read()
        txt = read_pdf_bytes(b) or ocr_pdf_bytes(b) if uploaded_file.filename.lower().endswith(".pdf") else ocr_image_bytes(b)
        if txt:
            upsert_text_chunks(txt, source="chat-attachment", link="", summary="User-attached document")
            file_context = f"User-attached document:\n{txt}"
    few_shot_question = search_chain.invoke({"query":body_message})

    index_context = ""
    if body_augment in ("index_only", "file_plus_index"):
        keywords = extract_keywords(few_shot_question or "")
        hits = pinecone_search_text(few_shot_question or "", top_k=2)

        weak = True
        for h in hits.get("result", {}).get("hits", []):
            if float(h.get("_score", 0.0)) >= SCORE_THRESHOLD:
                weak = False
                break

        if weak and keywords:
            await run_in_threadpool(arxiv_ingest_for_keywords, keywords, body_message or "")
            hits = pinecone_search_text(few_shot_question or "", top_k=3)

        index_context = build_context_from_hits(hits)

    if body_augment == "index_only":
        final_context = index_context + previous_context
        mode = "direct"
    else:  # file_plus_index
        has_file = bool(file_context.strip())
        if has_file:
            mode = "compress"
            final_context = "\n\n".join([c for c in (file_context) if c.strip()])
        else:
            mode = "direct"
            final_context = "\n\n".join([c for c in (index_context, previous_context) if c.strip()])

    reply = answer_with_context(body_message or "", user_id, final_context or "", temp, mode=mode)
    return ChatResponse(reply=reply)


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = read_pdf_bytes(pdf_bytes)
    if not text:
        text = ocr_pdf_bytes(pdf_bytes)
    added = upsert_text_chunks(text or "", source="upload-pdf")
    file.file.close()
    del file
    return IngestResponse(chunks=int(added))


@app.post("/ingest/image", response_model=IngestResponse)
async def ingest_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    text = ocr_image_bytes(img_bytes) if img_bytes else ""
    added = upsert_text_chunks(text or "", source="upload-image")
    return IngestResponse(chunks=int(added))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_research:app", host="0.0.0.0", port=8000, reload=True)
