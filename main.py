from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import time
import random
import io
import PyPDF2
import docx
import google.generativeai as genai
import pinecone
from sentence_transformers import SentenceTransformer
import logging
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
# Initialize Pinecone client
from pinecone import Pinecone
pinecone_index = None  # Global index holder

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pinecone_index

    # Warm Pinecone index once
    if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="aws-starter")
        )
        logger.info("Created new Pinecone index. Waiting for it to be ready...")
        time.sleep(10)

    pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
    logger.info("Pinecone index initialized.")
    yield
    logger.info("App shutdown")

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Processes large documents, performs contextual decisions, and provides explainable rationale.",
    version="1.0.0",
    lifespan=lifespan
)
import logging
from fastapi.responses import JSONResponse
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Processes large documents, performs contextual decisions, and provides explainable rationale.",
    version="1.0.0"
)
from fastapi import Request
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Invalid request: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid JSON format. Check your request body."},
    )

# Configuration
EMBEDDING_DIMENSION = 384
PINECONE_INDEX_NAME = "document-chunks"
MAX_RETRIES = 3
RETRY_DELAY = 1

# Initialize services
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
import os

pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

pinecone_client = Pinecone(api_key=pinecone_api_key)
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel('gemini-2.0-flash-lite')

# Helper Functions
def get_embedding(text: str) -> List[float]:
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate text embedding")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []
    paragraphs = text.split('\n\n')
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue
        sentences = paragraph.split('. ')
        current_chunk = []
        current_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_len = len(sentence)
            if current_len + sentence_len > chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                if overlap > 0:
                    current_chunk = current_chunk[-int(overlap/10):] if overlap else []
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            current_chunk.append(sentence)
            current_len += sentence_len + 2
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
    return chunks

def call_gemini_api_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_k": 40,
                    "candidate_count": 1,
                    "max_output_tokens": 2048,
                }
            )
            response.resolve()
            if not response.candidates or response.candidates[0].finish_reason != "STOP":
                logger.warning(f"Attempt {attempt + 1}: Invalid finish reason")
                time.sleep(RETRY_DELAY)
                continue
            if response.text:
                try:
                    clean_text = response.text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:-3].strip()
                    elif clean_text.startswith("```"):
                        clean_text = clean_text[3:-3].strip()
                    return json.loads(clean_text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    return {
                        "answer": "Could not process response format",
                        "conditions": [],
                        "rationale": "Error parsing LLM response",
                        "source_clauses_text": []
                    }
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    logger.error("All retries exhausted for Gemini API call")
    # Fallback response dictionary to avoid returning None.
    return {
        "answer": " valid response from LLM after multiple attempts.",
        "conditions": [],
        "rationale": "LLM response",
        "source_clauses_text": []
    }

def parse_document(file_content: bytes, filename: str) -> str:
    text = ""
    file_extension = filename.split('.')[-1].lower()
    try:
        if file_extension == "pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text() or ""
            except PyPDF2.PdfReadError as e:
                logger.error(f"PDF parsing error: {e}")
                raise HTTPException(status_code=400, detail=f"Error parsing PDF: {e}")
        elif file_extension == "docx":
            try:
                document = docx.Document(io.BytesIO(file_content))
                for para in document.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                logger.error(f"DOCX parsing error: {e}")
                raise HTTPException(status_code=400, detail=f"Error parsing DOCX: {e}")
        elif file_extension in ["txt", "md", "csv"]:
            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = file_content.decode('latin-1')
                except Exception as e:
                    logger.error(f"Text file decoding error: {e}")
                    raise HTTPException(status_code=400, detail="Could not decode text file")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
    except Exception as e:
        logger.error(f"Unexpected parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during document parsing: {e}")
    return text

def upsert_to_pinecone(doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
    try:
        # Obtain the index directly from the Pinecone client
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        vectors = [
            {
                "id": chunk["metadata"]["id"],
                "values": chunk["values"],
                "metadata": {
                    "doc_id": doc_id,
                    "text": chunk["metadata"]["text"],
                    **chunk["metadata"].get("additional_metadata", {})
                }
            }
            for chunk in chunks
        ]
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i + 100])
        return True
    except Exception as e:
        logger.error(f"Error upserting to Pinecone: {e}")
        return False
    
# Pydantic Models
class SourceClause(BaseModel):
    clause_id: str
    text: str
    doc_id: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = {}
from typing import List, Dict, Optional, Any
class QueryResponse(BaseModel):
    answer: str
    conditions: List[str]
    rationale: str
    source_clauses: List[SourceClause]
    pinecone_matches: Optional[List[Dict[str, Any]]] = None

class UnifiedQueryInput(BaseModel):
    query: Optional[str] = None
    queries: Optional[List[str]] = None
    top_k: int = 2
    include_summary: bool = False

    def get_queries(self) -> List[str]:
        if self.queries is not None and len(self.queries) > 0:
            return self.queries
        elif self.query is not None:
            return [self.query]
        else:
            raise ValueError("Either 'query' or 'queries' must be provided.")

class UnifiedQueryResponse(BaseModel):
    query_responses: List[QueryResponse]
    summary: Optional[str] = None

# API Endpoints
from fastapi.concurrency import run_in_threadpool

@app.post("/upload_document", response_model=Dict[str, str])
async def upload_document(file: UploadFile = File(...), doc_id: Optional[str] = None):
    if not doc_id:
        doc_id = f"doc_{int(time.time())}_{random.randint(1000, 9999)}"
    try:
        file_content = await file.read()
        extracted_text = await run_in_threadpool(parse_document, file_content, file.filename)

        text_chunks = await run_in_threadpool(chunk_text, extracted_text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted or chunked from the document.")

        ingestion_chunks = [
            {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": chunk_text_content
            }
            for i, chunk_text_content in enumerate(text_chunks)
        ]

        chunks_to_store = []

        for idx, chunk_data in enumerate(ingestion_chunks):
            try:
                embedding = await run_in_threadpool(get_embedding, chunk_data["text"])
                chunks_to_store.append({
                    "values": embedding,
                    "metadata": {
                        "id": chunk_data["chunk_id"],
                        "doc_id": doc_id,
                        "text": chunk_data["text"],
                        "additional_metadata": {
                            "original_filename": file.filename,
                            "file_type": file.content_type,
                            "upload_time": time.time()
                        }
                    }
                })
            except Exception as e:
                logger.error(f"Error processing chunk {idx}: {e}")
                continue

        success = await run_in_threadpool(upsert_to_pinecone, doc_id, chunks_to_store)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store document in vector database")

        return {
            "message": f"Document '{file.filename}' processed successfully.",
            "doc_id": doc_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during document processing: {e}")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi.concurrency import run_in_threadpool
import asyncio

@app.post("/unified_query", response_model=UnifiedQueryResponse)
async def unified_query(query_input: UnifiedQueryInput):
    queries = query_input.get_queries()
    pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)

    async def process_single_query(q: str) -> QueryResponse:
        try:
            query_embedding = await run_in_threadpool(get_embedding, q)
            pinecone_results = await run_in_threadpool(
                lambda: pinecone_index.query(
                    vector=query_embedding,
                    top_k=query_input.top_k,
                    include_metadata=True,
                    filter=None  # No document id filtering now.
                )
            )
            matches = pinecone_results.matches or []
            if not matches:
                return QueryResponse(
                    answer="No relevant information found",
                    conditions=[],
                    rationale="No matching content found",
                    source_clauses=[],
                    pinecone_matches=[]
                )

            trimmed_matches = [
                {
                    "id": m["id"],
                    "score": m.get("score"),
                    "metadata": {
                        "text": m["metadata"]["text"],
                        "doc_id": m["metadata"].get("doc_id", "")
                    }
                } for m in matches
            ]

            context_text = "\n\n".join(
                f"Clause ID: {m['id']}\nText: {format_clause_text(m['metadata']['text'])}"
                for m in matches
            )

            llm_prompt = f"""Analyze this query and context:
Query: {q}
Context: {context_text}
Provide JSON response with answer, conditions, rationale, and source clauses"""

            llm_response = await run_in_threadpool(call_gemini_api_with_retry, llm_prompt)

            return QueryResponse(
                answer=llm_response.get("answer", "No answer"),
                conditions=llm_response.get("conditions", []),
                rationale=llm_response.get("rationale", ""),
                source_clauses=[
                    SourceClause(
                        clause_id=m["id"],
                        text=format_clause_text(m["metadata"]["text"]),
                        doc_id=m["metadata"].get("doc_id", ""),
                        score=m.get("score")
                    ) for m in matches
                ],
                pinecone_matches=trimmed_matches
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResponse(
                answer=f"Error: {str(e)}",
                conditions=[],
                rationale="Processing failed",
                source_clauses=[],
                pinecone_matches=[]
            )

    # Run queries in parallel
    results = await asyncio.gather(*[process_single_query(q) for q in queries])
    summary = None

    if query_input.include_summary and len(results) > 1:
        summary_prompt = "Summarize these responses:\n" + "\n".join(
            f"Q: {q}\nA: {r.answer}" for q, r in zip(queries, results))
        summary_response = await run_in_threadpool(call_gemini_api_with_retry, summary_prompt)
        summary = summary_response.get("answer") if summary_response else None

    return UnifiedQueryResponse(query_responses=results, summary=summary)

# Add this helper function near your other helper functions, e.g., after chunk_text
def format_clause_text(text: str) -> str:
    """
    Break the input text into individual non-empty, trimmed lines and join them.
    This helps to align text properly.
    """
    # Split text by newline and punctuation if needed (here we split by newline)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Optionally, further split by punctuation and reassemble if required
    return "\n".join(lines)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

