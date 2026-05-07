class EnhancedDocumentStore:
    """
    Manages the complete document processing and retrieval pipeline.
    """

    def __init__(self):
        self.pages_info = []
        self.logical_docs = []
        self.chunks_metadata = []
        self.retriever = IntelligentRetriever()
        self.is_ready = False
        self.processing_stats = {}
        self.filename = None

    def process_pdf(self, pdf_file, filename: str = "document.pdf", use_llama_index: bool = True):
        """
        Complete PDF processing pipeline.
        """
        self.filename = filename
        self.is_ready = False
        start_time = datetime.now()

        try:
            # Extract and analyze PDF
            self.pages_info, self.logical_docs = extract_and_analyze_pdf(pdf_file)

            # Chunk documents with metadata
            self.chunks_metadata = process_all_documents(
                self.logical_docs,
                use_llama_index=use_llama_index
            )

            # Build retrieval indices
            self.retriever.build_indices(self.chunks_metadata)

            # Calculate processing statistics
            process_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats = {
                'filename': filename,
                'total_pages': len(self.pages_info),
                'documents_found': len(self.logical_docs),
                'total_chunks': len(self.chunks_metadata),
                'document_types': list(set(doc.doc_type for doc in self.logical_docs)),
                'processing_time': f"{process_time:.1f}s"
            }

            self.is_ready = True
            return True, self.processing_stats

        except Exception as e:
            return False, {'error': str(e)}

    def query(self, question: str, filter_type: Optional[str] = None,
             auto_route: bool = True, k: int = 2) -> Dict:
        """
        Query the document store.
        """
        if not self.is_ready:
            return {
                'answer': "Please upload and process a PDF first.",
                'sources': [],
                'confidence': 0.0
            }

        # Retrieve relevant chunks
        retrieved = self.retriever.retrieve(
            question, k=k,
            filter_doc_type=filter_type,
            auto_route=auto_route
        )

        # Generate answer with sources
        result = generate_answer_with_sources(
            question,
            retrieved,
            max_chunks=3,
            max_chars_per_chunk=600
        )

        return result

    def get_document_structure(self) -> List[Dict]:
        """
        Get the document structure for UI display.
        """
        if not self.logical_docs:
            return []

        structure = []
        for doc in self.logical_docs:
            structure.append({
                'id': doc.doc_id,
                'type': doc.doc_type,
                'pages': f"{doc.page_start + 1}-{doc.page_end + 1}",  # 1-indexed for UI
                'chunks': len(doc.chunks) if doc.chunks else 0,
                'preview': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            })

        return structure
