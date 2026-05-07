def generate_answer_with_sources(
    query: str,
    retrieved_chunks: List[Tuple[ChunkMetadata, float]],
    max_chunks: int = 3,
    max_chars_per_chunk: int = 600
) -> Dict:
    """
    Generate an answer with source attribution while keeping context small.

    Assumes retrieval scores are cosine-like similarity values.
    For display:
    - relative relevance = score / top_score
    - confidence = weighted top-score confidence
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information to answer your question.",
            "sources": [],
            "confidence": 0.0,
            "confidence_label": "Very Low"
        }

    # Keep top chunks only
    retrieved_chunks = retrieved_chunks[:max_chunks]

    # Raw similarity scores
    raw_scores = [max(0.0, float(score)) for _, score in retrieved_chunks]
    top_score = max(raw_scores) if raw_scores else 1.0

    context_parts = []
    sources = []

    for (chunk_meta, raw_score) in retrieved_chunks:
        trimmed_text = chunk_meta.text[:max_chars_per_chunk]

        # Relative score for display only
        relative_score = (raw_score / top_score) if top_score > 0 else 0.0
        relative_score = max(0.0, min(1.0, relative_score))

        context_parts.append(
            f"[From {chunk_meta.doc_type}, Pages {chunk_meta.page_start}-{chunk_meta.page_end}]"
        )
        context_parts.append(trimmed_text)
        context_parts.append("")

        sources.append({
            "doc_type": chunk_meta.doc_type,
            "pages": f"{chunk_meta.page_start}-{chunk_meta.page_end}",
            "relevance": f"{relative_score:.1%}",
            "relevance_label": score_label(relative_score),
            "raw_score": round(raw_score, 4),
            "preview": trimmed_text[:100] + "..."
        })

    context = "\n".join(context_parts)

    prompt = f"""You answer questions about pharmaceutical documents.

Use ONLY the context below.
Be concise.
If the answer is not clearly supported by the context, say that directly.

Context:
{context}

Question: {query}

Answer in 3-6 sentences max.
Mention the supporting document type and page range when relevant.
"""

    try:
        response = llm.complete(prompt)
        answer = response.text.strip()

        # Weighted confidence using raw similarity
        # This is better than averaging all chunk scores equally.
        if len(raw_scores) == 1:
            confidence_raw = raw_scores[0]
        elif len(raw_scores) == 2:
            confidence_raw = (raw_scores[0] * 0.7) + (raw_scores[1] * 0.3)
        else:
            confidence_raw = (
                raw_scores[0] * 0.6 +
                raw_scores[1] * 0.25 +
                raw_scores[2] * 0.15
            )

        # Normalize confidence relative to top score for friendlier display
        confidence = (confidence_raw / top_score) if top_score > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "confidence_label": score_label(confidence),
            "chunks_used": len(retrieved_chunks)
        }

    except Exception as e:
        print(f"Answer generation error: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": sources,
            "confidence": 0.0,
            "confidence_label": "Very Low"
        }


## Lookup helper

def is_lookup_query(query: str) -> bool:
    q = query.lower()
    lookup_terms = [
        "lot number", "batch number", "part number", "catalog number",
        "expiration date", "expiry date", "manufacture date",
        "sterilization", "gamma", "autoclave", "material",
        "iso", "supplier", "certificate", "page"
    ]
    return any(term in q for term in lookup_terms)


## Fast extractor

def generate_extract_answer(query: str,
                            retrieved_chunks: List[Tuple[ChunkMetadata, float]]) -> Dict:
    """
    Faster answer mode: return the most relevant retrieved text snippets
    without a full LLM generation call.
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information to answer your question.",
            "sources": [],
            "confidence": 0.0
        }

    lines = []
    sources = []
    display_scores = []

    for i, (chunk_meta, score) in enumerate(retrieved_chunks[:3], start=1):
        snippet = chunk_meta.text[:500]
        display_score = max(0.0, min(1.0, float(score)))
        display_scores.append(display_score)

        lines.append(
            f"Result {i} from {chunk_meta.doc_type} (Pages {chunk_meta.page_start}-{chunk_meta.page_end}):\n{snippet}"
        )
        sources.append({
            "doc_type": chunk_meta.doc_type,
            "pages": f"{chunk_meta.page_start}-{chunk_meta.page_end}",
            "relevance": f"{display_score:.2%}",
            "preview": snippet[:100] + "..."
        })

    avg_score = sum(display_scores) / len(display_scores) if display_scores else 0.0

    return {
        "answer": "\n\n".join(lines),
        "sources": sources,
        "confidence": avg_score,
        "chunks_used": min(len(retrieved_chunks), 3)
    }


