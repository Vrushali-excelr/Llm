# MedRAG-QA: Medical PDF RAG Question-Answering Model

A Retrieval-Augmented Generation (RAG) AI model that scans medical PDF documents and answers user questions based on their content.

## Overview
MedRAG-QA ingests medical documents like research papers, clinical guidelines, and patient reports. It builds a vector index from the PDFs, then uses a LLM + retriever pipeline to provide accurate, source-grounded answers to medical queries.

This project is for research/educational use only. It is not a substitute for professional medical advice.

## Key Features
- **PDF Ingestion**: Auto parse & chunk medical PDFs
- **Semantic Search**: FAISS vector store with embeddings for fast retrieval
- **RAG Pipeline**: Retrieves relevant context + generates answers with an LLM
- **Source Citations**: Responses include page/section references from the source PDF
- **QA Interface**: Simple FastAPI UI to ask questions

## Tech Stack
- **Language**: Python 3.10+
- **LLM**:  OpenAI GPT-4o-mini,
- **Embeddings**: [e.g. sentence-transformers/all-MiniLM-L6-v2]
- **Vector DB**: Pinecone
- **PDF Parsing**: PyPDF
- **Framework**: LangChain, FastAPI

## Quick Start

1. **Clone & Install**
```bash
git clone https://github.com/yourname/medrag-qa.git
cd medrag-qa
pip install -r requirements.txt
