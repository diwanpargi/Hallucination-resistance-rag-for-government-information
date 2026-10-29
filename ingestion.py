 
import os
import re
import shutil

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

import config


def setup_tesseract():
    """Windows pe Tesseract ka path PATH me jodta hai (OCR ke liye)."""
    os.environ["PATH"] += os.pathsep + config.TESSERACT_FOLDER
    print("Tesseract:", shutil.which("tesseract"))


def clean_text(text):
    """HTML tags remove and also remove whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_elements(pdf_path):
    """text extraction."""
    return partition_pdf(
        filename=pdf_path,
        strategy="auto",
        infer_table_structure=True,
    )


def clean_elements(elements):
    """clean every element."""
    cleaned = []
    for el in elements:
        text = clean_text(el.text)
        if text:
            el.text = text
            cleaned.append(el)
    return cleaned


def make_chunks(elements, filename):
    """make chunk by title ."""
    chunks = chunk_by_title(
        elements,
        max_characters=config.MAX_CHARS,
        new_after_n_chars=config.NEW_AFTER_CHARS,
        overlap=config.OVERLAP,
    )

    final_chunks = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata.to_dict()
        final_chunks.append({
            "chunk_id": f"{filename}____{i}",
            "text": chunk.text,
            "source": filename,                      
            "page_number": meta.get("page_number"),
        })
    return final_chunks


def ingest_pdf(pdf_path):
    """complete pdf ingestion: extract -> clean -> chunk. return Chunks list ."""
    filename = os.path.basename(pdf_path)

    elements = extract_elements(pdf_path)
    print("Extracted elements:", len(elements))

    elements = clean_elements(elements)
    print("After cleaning:", len(elements))
    if not elements:
        return []

    chunks = make_chunks(elements, filename)
    print("Chunks:", len(chunks))
    return chunks
