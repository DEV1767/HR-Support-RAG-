# HR Support RAG

An AI-powered HR Support Assistant built using Retrieval-Augmented Generation (RAG). The system retrieves relevant information from company HR documents and uses an LLM to generate answers based only on the retrieved context.

## How It Works

HR Documents
→ Document Loader
→ Text Splitting
→ Jina Embeddings
→ Qdrant Vector Database
→ Retriever
→ Guardials
→ Groq LLM
→ Guardials
→ AI HR Response
→ FastAPI

## Features

- PDF document loading and text extraction
- Document chunking using RecursiveCharacterTextSplitter
- Vector embeddings using Jina
- Vector storage and similarity search with Qdrant Cloud
- RAG-based question answering
- Groq LLM for response generation
- FastAPI API endpoint
- Streaming response support
- Retrieval and total response latency tracking
- Added Guardials for security 
- Monitoring of RAG using Langsmith

## Tech Stack

- Python
- LangChain
- Langsmith
- Jina Embeddings
- Qdrant Cloud
- Groq
- FastAPI
- UV

## API

### Chat

POST /api/chat

Request:

{
  "question": "What is the leave policy for the company?"
}

### Streaming Chat

POST /api/chat/stream

The streaming endpoint returns the AI response progressively.

## Project Structure

src/
├── api.py
├── embeddings.py
├── loader.py
├── model.py
├── retriver.py
├── Splitter.py
├── vectore_store.py
└── test.py

## Purpose

This project is currently a RAG prototype for an AI-based HR support system. The next stage is to integrate it with a Node.js backend and build a complete production-ready application.