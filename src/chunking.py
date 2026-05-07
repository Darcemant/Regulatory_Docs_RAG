def chunk_document_with_metadata(logical_doc: LogicalDocument,
                                chunk_size: int = 500,
                                overlap: int = 100) -> List[ChunkMetadata]:
    """
    Chunk a logical document while preserving rich metadata.
    Uses sliding window with overlap for better context.
    """
    chunks_metadata = []
    words = logical_doc.text.split()

    if len(words) <= chunk_size:
        # Document is small enough to be a single chunk
        chunk_meta = ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_0",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=0,
            page_start=logical_doc.page_start,
            page_end=logical_doc.page_end,
            text=logical_doc.text
        )
        chunks_metadata.append(chunk_meta)
    else:
        # Create overlapping chunks
        stride = chunk_size - overlap
        for i, start_idx in enumerate(range(0, len(words), stride)):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_text = ' '.join(words[start_idx:end_idx])

            # Calculate which pages this chunk spans
            # (simplified - in production, track more precisely)
            chunk_position = start_idx / len(words)
            page_range = logical_doc.page_end - logical_doc.page_start
            relative_page = int(chunk_position * page_range)
            chunk_page_start = logical_doc.page_start + relative_page
            chunk_page_end = min(chunk_page_start + 1, logical_doc.page_end)

            chunk_meta = ChunkMetadata(
                chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
                doc_id=logical_doc.doc_id,
                doc_type=logical_doc.doc_type,
                chunk_index=i,
                page_start=chunk_page_start,
                page_end=chunk_page_end,
                text=chunk_text
            )
            chunks_metadata.append(chunk_meta)

            if end_idx >= len(words):
                break

    return chunks_metadata


def chunk_with_llama_index(logical_doc: LogicalDocument,
                           chunk_size: int = 500,
                           chunk_overlap: int = 100) -> List[Document]:
    """
    Alternative: Use LlamaIndex's advanced chunking with metadata.
    """
    # Create LlamaIndex document with metadata
    doc = Document(
        text=logical_doc.text,
        metadata={
            "doc_id": logical_doc.doc_id,
            "doc_type": logical_doc.doc_type,
            "page_start": logical_doc.page_start,
            "page_end": logical_doc.page_end,
            "source": f"{logical_doc.doc_type}_document"
        }
    )

    # Use LlamaIndex's sentence splitter for better chunking
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        separator=" ",
    )

    # Create nodes (chunks) from document
    nodes = splitter.get_nodes_from_documents([doc])

    # Convert to our ChunkMetadata format for consistency
    chunks_metadata = []
    for i, node in enumerate(nodes):
        chunk_meta = ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=i,
            page_start=node.metadata.get("page_start", logical_doc.page_start),
            page_end=node.metadata.get("page_end", logical_doc.page_end),
            text=node.text
        )
        chunks_metadata.append(chunk_meta)

    return chunks_metadata


def process_all_documents(logical_docs: List[LogicalDocument],
                         use_llama_index: bool = True) -> List[ChunkMetadata]:
    """
    Process all logical documents into chunks with metadata.
    Can use either custom or LlamaIndex chunking.
    """
    all_chunks = []

    for logical_doc in logical_docs:
        if use_llama_index:
            chunks = chunk_with_llama_index(logical_doc)
        else:
            chunks = chunk_document_with_metadata(logical_doc)

        logical_doc.chunks = chunks  # Store reference
        all_chunks.extend(chunks)
        print(f"  {logical_doc.doc_type}: Created {len(chunks)} chunks")

    return all_chunks
