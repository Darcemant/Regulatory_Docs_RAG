# Pharmaceutical Document RAG System

This project is a Retrieval-Augmented Generation system designed to extract, index, retrieve, and answer questions from pharmaceutical-style PDF documents.

## Project Overview

The system processes PDF documents using OCR, metadata classification, text chunking, embedding generation, semantic retrieval, and LLM-based answer generation.

## Problem Statement

Pharmaceutical documentation is often long, technical, and difficult to search manually. This project explores how RAG can improve document search, source-backed question answering, and review efficiency.

## Key Features

- PDF ingestion
- EasyOCR-based text extraction
- Metadata classification
- Sliding-window text chunking
- SentenceTransformer embeddings
- FAISS vector retrieval
- LLM answer generation
- Source-backed responses
- Gradio interface

## Pipeline

1. Upload PDF documents
2. Extract text with OCR
3. Classify document/page metadata
4. Chunk text using sliding windows
5. Generate embeddings
6. Retrieve relevant chunks with FAISS
7. Generate grounded answers with an LLM
8. Display answer and sources in the interface

## Technical Stack

- Python
- EasyOCR
- PyTorch
- SentenceTransformers
- FAISS
- Phi-3 / local LLM
- Gradio
- pandas / NumPy

## Current Limitations

- Local LLM latency
- No persistent vector database
- No reranking layer
- Limited formal evaluation
- OCR quality depends on document scan quality

## Future Improvements

- Add ChromaDB or Qdrant
- Add reranking
- Add similarity thresholds
- Use API-based LLM for faster generation
- Build formal evaluation set
- Add persistent ingestion pipeline

## Repository Structure

```text
src/
notebooks/
docs/
reports/
assets/
