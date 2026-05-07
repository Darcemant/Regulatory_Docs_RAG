import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import numpy as np
import easyocr
import torch
import re

from typing import List, Dict

from src.schemas import PageInfo, LogicalDocument
from src.config import OCR_TRIGGER_MAX_CHARS

def extract_and_analyze_pdf(pdf_file) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    """
    Extract text from PDF and perform intelligent document analysis.

    Instrumented with per-stage timing so the bottleneck is visible.
    """
    print("Starting PDF extraction and analysis...")
    total_t0 = time.time()

    # --- 1. Extract text --------------------------------------------------
    stage_t0 = time.time()
    if isinstance(pdf_file, dict) and "content" in pdf_file:
        doc = fitz.open(stream=pdf_file["content"], filetype="pdf")
    elif hasattr(pdf_file, "read"):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_file)

    pages_info = []
    ocr_pages = 0
    ocr_time_total = 0.0

    for i, page in enumerate(doc):
        text = page.get_text()

        # Stricter OCR gate: only fire OCR when there's essentially no text.
        # Previously any non-whitespace char skipped OCR; now we require a
        # real threshold. Inverse side: if a page's text is stamps only, we
        # still OCR it. Tune OCR_TRIGGER_MAX_CHARS per your corpus.
        if len(text.strip()) < OCR_TRIGGER_MAX_CHARS:
            ocr_t0 = time.time()
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_data))
                ocr_results = easyocr_reader.readtext(np.array(img))
                text = " ".join([res[1] for res in ocr_results])
                ocr_time = time.time() - ocr_t0
                ocr_time_total += ocr_time
                ocr_pages += 1
                print(f"  Page {i}: OCR extracted {len(text)} chars in {ocr_time:.1f}s")
            except Exception as e:
                print(f"  Page {i}: OCR failed - {e}")
                text = ""

        pages_info.append(PageInfo(page_num=i, text=text))

    doc.close()

    if not pages_info:
        raise ValueError("No text could be extracted from PDF")

    extract_time = time.time() - stage_t0
    print(f"[TIMING] Extraction: {extract_time:.1f}s total "
          f"({len(pages_info)} pages, {ocr_pages} needed OCR, "
          f"OCR time: {ocr_time_total:.1f}s)")

    # --- 2. Classify every page in a single batched pass ------------------
    stage_t0 = time.time()
    batch_classify_pages(pages_info)
    classify_time = time.time() - stage_t0
    print(f"[TIMING] Classification: {classify_time:.1f}s")

    # --- 3. Group into logical documents with heuristic-only boundaries --
    stage_t0 = time.time()
    logical_docs = []
    doc_counter = 0

    pages_info[0].page_in_doc = 0
    current_doc_pages = [pages_info[0]]
    print(f"  Page 0: New document - {pages_info[0].doc_type}")

    for i in range(1, len(pages_info)):
        prev_page = pages_info[i - 1]
        curr_page = pages_info[i]

        if is_same_document(prev_page, curr_page):
            curr_page.doc_type = current_doc_pages[0].doc_type
            curr_page.page_in_doc = len(current_doc_pages)
            current_doc_pages.append(curr_page)
        else:
            logical_docs.append(LogicalDocument(
                doc_id=f"doc_{doc_counter}",
                doc_type=current_doc_pages[0].doc_type,
                page_start=current_doc_pages[0].page_num,
                page_end=current_doc_pages[-1].page_num,
                text="\n\n".join([p.text for p in current_doc_pages]),
            ))
            doc_counter += 1
            curr_page.page_in_doc = 0
            current_doc_pages = [curr_page]
            print(f"  Page {i}: New document - {curr_page.doc_type}")

    if current_doc_pages:
        logical_docs.append(LogicalDocument(
            doc_id=f"doc_{doc_counter}",
            doc_type=current_doc_pages[0].doc_type,
            page_start=current_doc_pages[0].page_num,
            page_end=current_doc_pages[-1].page_num,
            text="\n\n".join([p.text for p in current_doc_pages]),
        ))

    group_time = time.time() - stage_t0
    print(f"[TIMING] Grouping: {group_time:.1f}s")
    print(f"[TIMING] Total extract_and_analyze_pdf: {time.time() - total_t0:.1f}s")
    print(f"Identified {len(logical_docs)} logical documents")
    for ld in logical_docs:
        print(f"   - {ld.doc_type}: Pages {ld.page_start}-{ld.page_end}")

    return pages_info, logical_docs
