def predict_query_document_type(query: str) -> Tuple[str, float]:
    """
    Fast keyword-based query router.
    Predicts which document type is most likely to contain the answer.
    """

    DOC_TYPE_KEYWORDS = {
        "Certificate Of Quality": [
            "lot", "batch", "certificate", "coa", "expiration",
            "manufacture", "manufactured", "test result", "sterility"
        ],
        "Packaging Specification": [
            "packaging", "carton", "label", "component",
            "part number", "dimensions", "specification"
        ],
        "Bse/Tse Declaration": [
            "bse", "tse", "animal", "origin", "declaration"
        ],
        "Material Description": [
            "material", "composition", "construction",
            "sterilization", "compatibility"
        ],
        "Supplier Qualification": [
            "supplier", "audit", "iso", "qualification", "approved"
        ],
        "Chain Of Custody": [
            "traceability", "custody", "shipment", "assembly", "flow"
        ],
        "Cover Letter": [
            "letter", "storage", "formal", "product information"
        ]
    }

    q = query.lower()
    scores = {}

    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return "Other", 0.0

    best_doc_type = max(scores, key=scores.get)
    confidence = min(scores[best_doc_type] / 3.0, 1.0)

    return clean_doc_type(best_doc_type), confidence

def infer_doc_type_from_query(query: str) -> str | None:
    q = query.lower()

    if any(term in q for term in ["lot number", "expiration", "expiry", "exp date", "mfg date", "article number", "certificate", "quality"]):
        return "Certificate Of Quality"

    if any(term in q for term in ["packaging", "configuration", "component", "dimensions", "drawing", "revision", "specification"]):
        return "Packaging Specification"

    if any(term in q for term in ["bse", "tse", "animal origin"]):
        return "Bse/Tse Declaration"

    if any(term in q for term in ["material", "composition", "construction", "physical properties", "compatibility"]):
        return "Material Description"

    if any(term in q for term in ["supplier", "audit", "iso", "qualification", "approved supplier"]):
        return "Supplier Qualification"

    if any(term in q for term in ["custody", "traceability", "shipment flow", "assembly record"]):
        return "Chain Of Custody"

    if any(term in q for term in ["cover letter", "to whom it may concern", "letter"]):
        return "Cover Letter"

    return None

class IntelligentRetriever:
    """
    Advanced retrieval system with metadata filtering and query routing.
    """

    def __init__(self):
        self.index = None
        self.chunks_metadata = []
        self.doc_type_indices = {}  # Separate indices per doc type

    def build_indices(self, chunks_metadata: List[ChunkMetadata]):
        """
        Build FAISS indices with document type segregation.
        Uses normalized embeddings + inner product so returned scores behave
        like cosine similarity.
        """
        print("Building vector indices...")
        self.chunks_metadata = chunks_metadata
        self.doc_type_indices = {}

        # Create embeddings for all chunks
        texts = [chunk.text for chunk in chunks_metadata]
        embeddings = embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")

        # Normalize embeddings so IndexFlatIP approximates cosine similarity
        faiss.normalize_L2(embeddings)

        # Store embeddings in metadata
        for i, chunk in enumerate(chunks_metadata):
            chunk.embedding = embeddings[i]

        # Build main index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Build separate indices for each document type
        doc_types = set(chunk.doc_type for chunk in chunks_metadata)
        for doc_type in doc_types:
            type_indices = [
                i for i, chunk in enumerate(chunks_metadata)
                if chunk.doc_type == doc_type
            ]
            if type_indices:
                type_embeddings = embeddings[type_indices].copy()
                type_index = faiss.IndexFlatIP(dim)
                type_index.add(type_embeddings)
                self.doc_type_indices[doc_type] = {
                    "index": type_index,
                    "mapping": type_indices  # Maps back to original chunks
                }

        print(f"Indexed {len(chunks_metadata)} chunks across {len(doc_types)} document types")

    def retrieve(self, query: str, k: int = 4,
                 filter_doc_type: Optional[str] = None,
                 auto_route: bool = True) -> List[Tuple[ChunkMetadata, float]]:
        """
        Retrieve relevant chunks with optional filtering and routing.
        Returns chunks with cosine-like similarity scores.
        """
        query_embedding = embed_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        # Determine which index to search
        if filter_doc_type and filter_doc_type in self.doc_type_indices:
            type_data = self.doc_type_indices[filter_doc_type]
            D, I = type_data["index"].search(query_embedding, k)
            chunk_indices = [type_data["mapping"][i] for i in I[0] if i != -1]
            scores = [float(d) for i, d in zip(I[0], D[0]) if i != -1]

        elif auto_route:
            predicted_type = infer_doc_type_from_query(query)
            print(f"Query routed to: {predicted_type}")

            if predicted_type and predicted_type in self.doc_type_indices:
                type_data = self.doc_type_indices[predicted_type]
                D, I = type_data["index"].search(query_embedding, k)
                chunk_indices = [type_data["mapping"][i] for i in I[0] if i != -1]
                scores = [float(d) for i, d in zip(I[0], D[0]) if i != -1]
            else:
                D, I = self.index.search(query_embedding, k)
                chunk_indices = [i for i in I[0] if i != -1]
                scores = [float(d) for i, d in zip(I[0], D[0]) if i != -1]

        else:
            D, I = self.index.search(query_embedding, k)
            chunk_indices = [i for i in I[0] if i != -1]
            scores = [float(d) for i, d in zip(I[0], D[0]) if i != -1]

        results = []
        for idx, i in enumerate(chunk_indices):
            # Clip only for display/reporting safety. Retrieval already used raw scores.
            score = max(0.0, min(1.0, scores[idx]))
            results.append((self.chunks_metadata[i], score))

        return results
