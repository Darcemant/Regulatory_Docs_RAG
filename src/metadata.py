import re
import json

from typing import List, Dict, Tuple, Optional

from src.schemas import PageInfo, LogicalDocument

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extract_page_signals(text: str) -> dict:
    """Extract cheap regex-based signals used for heuristic boundary detection."""
    raw = text or ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    top_lines = lines[:12]
    top_blob = " ".join(top_lines)

    title_candidates = []
    for ln in top_lines:
        alpha = sum(ch.isalpha() for ch in ln)
        upper = sum(ch.isupper() for ch in ln if ch.isalpha())
        ratio = upper / alpha if alpha else 0

        score = 0
        if 5 <= len(ln) <= 80:
            score += 1
        if ratio > 0.6:
            score += 1
        if any(term in ln.lower() for term in [
            "certificate", "specification", "declaration",
            "description", "qualification", "custody"
        ]):
            score += 2

        if score > 0:
            title_candidates.append((score, ln))

    title = max(title_candidates, default=(0, ""))[1].lower()

    doc_no_match = re.search(
        r"(document\s*(no|number)?|doc\s*(no|number)?|reference|ref|specification|spec|part\s*number|article\s*number|certificate\s*number|lot\s*number)\s*[:#]?\s*([A-Z0-9\-/\.]+)",
        top_blob, re.I
    )

    page_no_match = re.search(r"page\s+(\d+)\s*(of|/)\s*(\d+)", top_blob, re.I)
    continued = bool(re.search(r"\b(continued|cont\'d)\b", top_blob, re.I))

    return {
        "title": title,
        "doc_no": doc_no_match.group(4).lower() if doc_no_match else None,
        "page_no": int(page_no_match.group(1)) if page_no_match else None,
        "page_total": int(page_no_match.group(3)) if page_no_match else None,
        "continued": continued,
        "top_blob": top_blob.lower(),
    }

def heuristic_doc_type(text: str) -> Optional[str]:
    """Fast keyword-based classifier. Returns None when unsure so the caller
    can defer to the batched LLM call."""
    t = (text or "").lower()

    if any(x in t for x in ["certificate of quality", "certificate of analysis", "lot number", "batch number"]):
        return "Certificate Of Quality"

    if any(x in t for x in ["packaging specification", "carton", "blister", "label", "dimensions"]):
        return "Packaging Specification"

    if any(x in t for x in ["bse/tse", "bovine spongiform", "tse declaration", "animal origin"]):
        return "Bse/Tse Declaration"

    if any(x in t for x in ["material description", "composition", "sterilization method", "physical properties"]):
        return "Material Description"

    if any(x in t for x in ["supplier qualification", "approved supplier", "audit history"]):
        return "Supplier Qualification"

    if any(x in t for x in ["chain of custody", "traceability", "custody"]):
        return "Chain Of Custody"

    if any(x in t for x in ["to whom it may concern", "dear sir", "dear madam", "sincerely"]):
        return "Cover Letter"

    return None

def clean_doc_type(response):
    """Clean up LLM response to extract a valid doc_type label."""
    cleaned = response.strip().replace('"', '').replace('`', '').replace('*', '').lower().replace(".", "").strip()
    cleaned_title = cleaned.title()
    for label in VALID_DOC_TYPES:
        if label.lower() in cleaned.lower():
            return label
    return cleaned_title

def classify_document_type(text: str, max_length: int = 1500) -> str:
    """
    Classify the document type based on its content.
    Uses LLM to intelligently identify pharmaceutical document category.
    """
    # Truncate text if too long to avoid token limits
    text_sample = text[:max_length] if len(text) > max_length else text

    prompt = f"""You are a pharmaceutical document classifier. Based on the page
content below, classify it into ONE of these document types:

- Cover Letter: A formal letter (often starting with "To Whom It
  May Concern") discussing product information or storage conditions.
- Certificate of Quality: Contains lot numbers, manufacture dates,
  expiration dates, and test results (autoclave, gamma irradiation).
- Packaging Specification: Describes packaging components, materials,
  part numbers, and configuration change history.
- BSE/TSE Declaration: A declaration about animal-origin materials
  and transmissible spongiform encephalopathy compliance.
- Material Description: Lists materials of construction, sterilization
  compatibility, and physical properties of a product.
- Supplier Qualification: Contains supplier audit history,
  certifications (ISO 9001, ISO 13485), and approved product lists.
- Chain of Custody: Lists manufactured assemblies, traceability
  information, and the manufacturing-to-shipment flow.
- Other: Use ONLY if the content does not match any of the above.

Page content:
{text_sample}

Respond with ONLY the document type name. No explanation."""

    try:
        response = llm.complete(prompt)
        return clean_doc_type(response.text)
    except Exception as e:
        print(f"Classification error: {e}")
        return 'Other'

  def is_same_document(prev_page, curr_page) -> bool:
    """
    Heuristic-only boundary detection. NO LLM call.

    Strategy:
      1. If the pre-computed doc_types differ, it's a new document.
      2. Otherwise, look at regex signals (doc_no, title, "continued", "page X of Y").
      3. If signals are inconclusive -> default to False (new doc).

    Per the mentor: retrieval handles over-splitting well; paying per-page LLM
    calls just to avoid a few extra splits is a bad trade.
    """
    if not prev_page.text or not curr_page.text:
        return False

    # Doc-type mismatch is a hard signal that these are different logical docs.
    if prev_page.doc_type and curr_page.doc_type and prev_page.doc_type != curr_page.doc_type:
        return False

    prev_signals = extract_page_signals(prev_page.text)
    curr_signals = extract_page_signals(curr_page.text)

    verdict = likely_same_document(prev_signals, curr_signals)
    if verdict is None:
        # Uncertain -> default to new doc (mentor's call).
        return False
    return verdict

def likely_same_document(prev_signals: dict, curr_signals: dict) -> Optional[bool]:
    """Pure-signal verdict. Returns True/False when confident, None when unsure."""
    pos = 0
    neg = 0

    if prev_signals["doc_no"] and curr_signals["doc_no"]:
        if prev_signals["doc_no"] == curr_signals["doc_no"]:
            pos += 3
        else:
            neg += 3

    if curr_signals["continued"]:
        pos += 2

    if curr_signals["page_no"] and curr_signals["page_no"] > 1:
        pos += 1

    if prev_signals["title"] and curr_signals["title"]:
        if prev_signals["title"] == curr_signals["title"]:
            pos += 2
        else:
            neg += 1

    if pos >= 3 and neg == 0:
        return True
    if neg >= 3 and pos == 0:
        return False

    return None

def _build_batch_prompt(batch_entries):
    """Build a prompt asking the LLM to classify multiple pages in one shot."""
    labeled = []
    for page_idx, snippet in batch_entries:
        labeled.append(f"--- PAGE {page_idx} ---\n{snippet}")
    pages_block = "\n\n".join(labeled)

    return f"""You are a pharmaceutical document classifier.

For each page below, choose EXACTLY ONE label from this list:
- Cover Letter
- Certificate Of Quality
- Packaging Specification
- Bse/Tse Declaration
- Material Description
- Supplier Qualification
- Chain Of Custody
- Other

Return ONLY a JSON array. No prose, no explanation, no markdown fences.
Format: [{{"page": <int>, "type": "<label>"}}, ...]

Pages:
{pages_block}
"""

def _parse_batch_response(response_text):
    """Extract a JSON array from a (possibly messy) LLM response."""
    # Strip common markdown fences
    cleaned = response_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")

    # Find the first [...] block
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        data = _json.loads(match.group(0))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def batch_classify_pages(pages_info):
    """
    Classify every page in `pages_info` in place, setting `page.doc_type`.

    Strategy:
      1. Run the fast heuristic over every page.
      2. Collect the pages where the heuristic returned None.
      3. Make ONE batched LLM call (chunked into BATCH_SIZE-sized groups for
         context-window safety) to classify the leftovers.
      4. Anything still unclassified after that falls back to "Other".

    For an N-page PDF with M heuristically-unclassified pages, this costs
    ceil(M / BATCH_SIZE) LLM calls instead of N+ calls.
    """
    uncertain = []
    for i, page in enumerate(pages_info):
        dt = heuristic_doc_type(page.text)
        if dt is not None:
            page.doc_type = dt
        else:
            uncertain.append(i)

    if not uncertain:
        print(f"  Heuristic classified all {len(pages_info)} pages. 0 LLM calls needed.")
        return

    print(f"  Heuristic classified {len(pages_info) - len(uncertain)}/{len(pages_info)} pages. "
          f"Batching {len(uncertain)} remaining pages into LLM calls.")

    # Chunk into BATCH_SIZE-sized groups
    batches = [uncertain[i:i + BATCH_SIZE] for i in range(0, len(uncertain), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches, start=1):
        entries = [(idx, normalize_text(pages_info[idx].text)[:SNIPPET_CHARS]) for idx in batch]
        prompt = _build_batch_prompt(entries)

        try:
            response = llm.complete(prompt)
            parsed = _parse_batch_response(response.text)
        except Exception as e:
            print(f"    Batch {batch_num} LLM call failed: {e}")
            parsed = []

        # Apply the labels from this batch
        for item in parsed:
            idx = item.get("page")
            raw_label = item.get("type", "Other")
            if isinstance(idx, int) and 0 <= idx < len(pages_info) and pages_info[idx].doc_type is None:
                label = clean_doc_type(raw_label)
                pages_info[idx].doc_type = label if label in VALID_DOC_TYPES else "Other"

        print(f"    Batch {batch_num}/{len(batches)} done ({len(batch)} pages).")

    # Anything left -> Other
    for idx in uncertain:
        if pages_info[idx].doc_type is None:
            pages_info[idx].doc_type = "Other"
