# Import necessary libraries
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import json
import time
import random
import io
import PyPDF2
import docx
import google.generativeai as genai  # Updated import

# --- FastAPI Application Setup ---
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Processes large documents, performs contextual decisions, and provides explainable rationale.",
    version="1.0.0"
)

# --- Configuration ---
EMBEDDING_DIMENSION = 1536

# Configure Google Generative AI
GEMINI_API_KEY = ''  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Create model instance with Gemini 2.5 Flash
model = genai.GenerativeModel('gemini-2.5-flash')  # Updated model name

# --- Mock Document Store ---
MOCK_DOCUMENT_STORE = {}


# --- Helper Functions ---

# Placeholder for embedding generation.
# In a real system, this would call an actual embedding model (e.g., from Cohere, OpenAI, or a local model).
def get_embedding(text: str) -> List[float]:
    """
    Generates a mock embedding for the given text.
    In a real application, this would call an embedding model API.
    """
    # Simulate a fixed-size embedding vector
    # For demonstration, we'll just use a simple hash-based "embedding"
    # A real embedding would be a dense vector capturing semantic meaning.
    random.seed(hash(text) % (2**32 - 1)) # Seed for consistent mock embeddings
    return [random.random() for _ in range(EMBEDDING_DIMENSION)] # Use defined dimension

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of a specified size with optional overlap.
    A more sophisticated chunking strategy might be needed for production.
    """
    chunks = []
    if not text:
        return chunks

    words = text.split()
    current_chunk = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            current_chunk = current_chunk[-(overlap // 5):] if overlap else [] # Simple word-based overlap
            current_len = sum(len(w) for w in current_chunk) + len(current_chunk) - 1 if current_chunk else 0

        current_chunk.append(word)
        current_len += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def call_gemini_api(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Uses Gemini 2.5 Flash for response generation with structured output.
    """
    try:
        # Format the prompt to encourage structured JSON output
        formatted_prompt = f"""
        Please provide your response in valid JSON format with the following structure:
        {{
            "answer": "A clear and direct answer to the query",
            "conditions": ["condition 1", "condition 2", ...],
            "rationale": "Your reasoning based on the provided clauses",
            "source_clauses_text": ["full text of relevant clause 1", "full text of relevant clause 2", ...]
        }}

        Query and Context:
        {prompt}
        """
        
        response = model.generate_content(
            formatted_prompt,
            generation_config={
                "temperature": 0.3,
                "top_k": 40,
                "candidate_count": 1,
                "max_output_tokens": 1024,
            }
        )
        
        # Wait for response to complete
        response.resolve()
        
        if response.text:
            try:
                # Clean the response text to ensure valid JSON
                clean_text = response.text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:-3]  # Remove ```json and ``` markers
                elif clean_text.startswith("```"):
                    clean_text = clean_text[3:-3]  # Remove ``` markers
                
                return json.loads(clean_text)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from response: {response.text}")
                print(f"JSON Error: {e}")
                # Fallback response structure
                return {
                    "answer": "Could not process response format",
                    "conditions": [],
                    "rationale": "Error parsing LLM response",
                    "source_clauses_text": []
                }
        return None
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def parse_document(file_content: bytes, filename: str) -> str:
    """
    Parses the content of a document (PDF, DOCX, or plain text) and returns its text.
    """
    text = ""
    file_extension = filename.split('.')[-1].lower()

    if file_extension == "pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() or ""
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing PDF: {e}")
    elif file_extension == "docx":
        try:
            document = docx.Document(io.BytesIO(file_content))
            for para in document.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing DOCX: {e}")
    elif file_extension in ["txt", "eml"]: # Basic support for text and email files
        try:
            text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            text = file_content.decode('latin-1') # Fallback for common encodings
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}. Only PDF, DOCX, TXT, EML are supported.")
    return text


# --- Pydantic Models for Request and Response ---

class DocumentInput(BaseModel):
    """
    Model for ingesting a document.
    For simplicity, we assume text is already extracted and chunked when this is used directly.
    """
    doc_id: str
    chunks: List[Dict[str, str]] # List of {"chunk_id": "...", "text": "..."}
    metadata: Optional[Dict[str, Any]] = {}

class QueryInput(BaseModel):
    """
    Model for a user query.
    """
    query: str
    doc_id: Optional[str] = None # Optional: query a specific document
    top_k: int = 5 # Number of top similar clauses to retrieve

class SourceClause(BaseModel):
    """
    Model for a retrieved source clause.
    """
    clause_id: str
    text: str
    doc_id: str
    # Add other relevant metadata if available
    page_number: Optional[int] = int
    section_title: Optional[str] = str

class QueryResponse(BaseModel):
    """
    Model for the structured JSON response.
    """
    answer: str
    conditions: List[str]
    rationale: str
    source_clauses: List[SourceClause]

# --- API Endpoints ---

@app.post("/ingest_document", response_model=Dict[str, str], include_in_schema=False)
async def ingest_document(doc_input: DocumentInput):
    """
    (Internal endpoint) Ingests a document by processing its chunks and storing them.
    """
    doc_id = doc_input.doc_id
    chunks_to_store = []

    for idx, chunk_data in enumerate(doc_input.chunks):
        chunk_id = chunk_data.get("chunk_id", f"{doc_id}_chunk_{idx}")
        text = chunk_data.get("text")
        if not text:
            print(f"Warning: Chunk {chunk_id} for doc {doc_id} has no text and will be skipped.")
            continue

        embedding = get_embedding(text)
        chunks_to_store.append({
            "values": embedding,
            "metadata": {
                "id": chunk_id,  # Add id to metadata
                "doc_id": doc_id,
                "text": text,
                **doc_input.metadata  # Include document-level metadata
            }
        })

    # --- Mock In-memory Storage (Primary storage as per your request) ---
    MOCK_DOCUMENT_STORE[doc_id] = {
        "chunks": chunks_to_store,
        "raw_chunks_data": doc_input.chunks # Store original chunks for retrieval
    }
    print(f"Document '{doc_id}' ingested to mock store with {len(chunks_to_store)} chunks.")

    return {"message": f"Document '{doc_id}' ingested successfully."}


@app.post("/upload_document", response_model=Dict[str, str])
async def upload_document(file: UploadFile = File(...), doc_id: Optional[str] = None):
    """
    Uploads a document (PDF, DOCX, TXT, EML), extracts text, chunks it,
    and ingests it into the system's mock store.
    """
    if not doc_id:
        # Generate a unique doc_id if not provided
        doc_id = f"doc_{int(time.time())}_{random.randint(1000, 9999)}"

    try:
        file_content = await file.read()
        extracted_text = parse_document(file_content, file.filename)
        text_chunks = chunk_text(extracted_text)

        if not text_chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted or chunked from the document.")

        # Prepare chunks for ingestion
        ingestion_chunks = []
        for i, chunk_text_content in enumerate(text_chunks):
            ingestion_chunks.append({
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": chunk_text_content
            })

        # Call the internal ingest_document function
        await ingest_document(DocumentInput(
            doc_id=doc_id,
            chunks=ingestion_chunks,
            metadata={"original_filename": file.filename, "file_type": file.content_type}
        ))

        return {"message": f"Document '{file.filename}' (ID: {doc_id}) uploaded and processed successfully."}

    except HTTPException as e:
        raise e # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during document processing: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_system(query_input: QueryInput):
    """
    Enhanced query processing with better context handling and structured responses.
    """
    user_query = query_input.query
    doc_id_filter = query_input.doc_id
    top_k = query_input.top_k

    query_embedding = get_embedding(user_query)
    retrieved_clauses = []

    # --- Semantic Search (Mocked In-memory) ---
    all_chunks_to_search = []
    if doc_id_filter:
        if doc_id_filter not in MOCK_DOCUMENT_STORE:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id_filter}' not found in mock store.")
        all_chunks_to_search = MOCK_DOCUMENT_STORE[doc_id_filter]["chunks"]
    else:
        # Search across all ingested documents
        for doc_data in MOCK_DOCUMENT_STORE.values():
            all_chunks_to_search.extend(doc_data["chunks"])

    # Simple cosine similarity for mock search
    def cosine_similarity(vec1, vec2):
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = sum(v**2 for v in vec1)**0.5
        magnitude2 = sum(v**2 for v in vec2)**0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    similarities = []
    for chunk in all_chunks_to_search:
        similarity = cosine_similarity(query_embedding, chunk["values"])
        similarities.append((similarity, chunk["metadata"]))

    # Sort by similarity in descending order and get top_k
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_similar_chunks_metadata = [s[1] for s in similarities[:top_k]]

    for chunk_meta in top_similar_chunks_metadata:
        retrieved_clauses.append(SourceClause(
            clause_id=chunk_meta["id"],
            text=chunk_meta["text"],
            doc_id=chunk_meta["doc_id"],
            page_number=chunk_meta.get("page_number"),
            section_title=chunk_meta.get("section_title")
        ))
    print(f"Retrieved {len(retrieved_clauses)} clauses from mock store.")


    if not retrieved_clauses:
        return QueryResponse(
            answer="I could not find any relevant information in the documents.",
            conditions=[],
            rationale="No relevant clauses were retrieved based on your query.",
            source_clauses=[]
        )

    # --- LLM Contextual Decision Making ---
    # Construct the prompt for the LLM
    context_text = "\n\n".join([f"Clause ID: {c.clause_id}\nText: {c.text}" for c in retrieved_clauses])
    llm_prompt = f"""
    Context: You are analyzing documents in {query_input.doc_id if query_input.doc_id else 'all documents'}.
    Query: {query_input.query}

    Relevant clauses:
    {context_text}

    Provide a structured response with:
    1. Direct answer
    2. Specific conditions
    3. Clear rationale with clause references
    4. Source clause details
    
    Format as JSON with schema:
    {{
        "answer": "string",
        "conditions": ["string"],
        "rationale": "string",
        "source_clauses": [{{
            "clause_id": "string",
            "text": "string",
            "relevance": "string"
        }}]
    }}
    """

    # Define the JSON schema for the LLM response
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "answer": {"type": "STRING", "description": "Direct answer to the user's query."},
            "conditions": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "List of specific conditions or details related to the answer."
            },
            "rationale": {
                "type": "STRING",
                "description": "Explanation of how the answer was derived, referencing relevant Clause IDs."
            },
            "source_clauses_text": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "List of the full text of the most relevant source clauses."
            }
        },
        "required": ["answer", "conditions", "rationale", "source_clauses_text"]
    }

    llm_response_data = call_gemini_api(llm_prompt)

    if not llm_response_data:
        raise HTTPException(status_code=500, detail="Failed to get a valid response from the LLM.")

    return QueryResponse(
        answer=llm_response_data.get("answer", "No answer found."),
        conditions=llm_response_data.get("conditions", []),
        rationale=llm_response_data.get("rationale", "No rationale provided."),
        source_clauses=retrieved_clauses
    )

# --- Example Usage (for testing locally) ---
# To run this:
# 1. Save as `main.py`
# 2. Install uvicorn, fastapi, requests, pydantic, PyPDF2, python-docx:
#    `pip install uvicorn fastapi requests pydantic PyPDF2 python-docx`
# 3. Run: `uvicorn main:app --reload`
# 4. Access API docs at `http://127.0.0.1:8000/docs`

# Example of how you might upload a document via API (using http://127.0.0.1:8000/docs):
# Go to /upload_document endpoint, click "Try it out", and use the "Choose File" button.
# You can optionally provide a doc_id in the text field.

# Example of how you might query via API (using http://127.0.0.1:8000/docs):
# POST /query
# {
#   "query": "Does this policy cover knee surgery,  and what are the conditions?",
#   "doc_id": "policy_123", # Use the doc_id returned from /upload_document (if filtering)
#   "top_k": 3
# }
