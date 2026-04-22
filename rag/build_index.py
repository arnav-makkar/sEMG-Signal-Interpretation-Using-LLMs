#!/usr/bin/env python3
"""
Build ChromaDB Vector Index from Research Paper PDFs
=====================================================
Extracts text from PDFs in Activation/, chunks it, embeds with
sentence-transformers, and stores in ChromaDB for semantic retrieval.

Usage:
    python3 rag/build_index.py            # build (skip if exists)
    python3 rag/build_index.py --force     # rebuild from scratch
"""

import os
import re
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# ==============================
# CONFIG
# ==============================

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Activation')
VECTORSTORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectorstore')
COLLECTION_NAME = 'emg_papers'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 300       # words
CHUNK_OVERLAP = 50     # words

PDF_METADATA = {
    "Konrad_2005_ABC_of_EMG.pdf": {
        "short_name": "Konrad2005",
        "title": "The ABC of EMG: A Practical Introduction to Kinesiological Electromyography",
        "tags": "amplitude,electrode,quality,baseline,practical,activation,normalization"
    },
    "Nazmi_2016_EMG_Classification_Review.pdf": {
        "short_name": "Nazmi2016",
        "title": "A Review of Classification Techniques of EMG Signals during Isotonic and Isometric Contractions",
        "tags": "classification,gesture,features,WAMP,motor_units,activation,onset"
    },
    "Phinyomark_2012_MNF_MDF_IntechOpen.pdf": {
        "short_name": "Phinyomark2012",
        "title": "Mean and Median Frequency of EMG Signal",
        "tags": "frequency,MNF,MDF,fatigue,spectral,features"
    },
    "Vigotsky_2018_Signal_Amplitudes_sEMG.pdf": {
        "short_name": "Vigotsky2018",
        "title": "Interpreting Signal Amplitudes in Surface EMG Studies in Sport and Rehabilitation Sciences",
        "tags": "amplitude,normalization,MVC,force,interpretation,activation"
    },
    "Yang_2024_EMGBench_NeurIPS.pdf": {
        "short_name": "Yang2024",
        "title": "EMGBench: Benchmarking Out-of-Distribution Generalization and Adaptation for Electromyography",
        "tags": "benchmark,generalization,fatigue,gesture,spatial,dataset,activation"
    },
}


# ==============================
# PDF TEXT EXTRACTION
# ==============================

def extract_text_from_pdf(pdf_path: str) -> list:
    """Extract text per page from a PDF. Returns list of (page_num, text)."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text = _clean_text(text)
            pages.append((i + 1, text))
    return pages


def _clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Fix common PDF extraction artifacts
    text = re.sub(r'\s+', ' ', text)           # collapse whitespace
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # fix broken hyphenation
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # remove standalone page numbers
    text = text.strip()
    return text


# ==============================
# CHUNKING
# ==============================

def detect_sections(full_text: str) -> list:
    """
    Try to split text into sections by detecting headings.
    Returns list of (section_name, section_text).
    """
    # Common heading patterns in academic papers
    patterns = [
        r'(?:^|\n)(\d+\.?\d*\.?\s+[A-Z][A-Za-z\s:,]+?)(?=\n)',    # "1.2 Introduction"
        r'(?:^|\n)([A-Z][A-Z\s]{4,40})(?=\n)',                       # "INTRODUCTION"
    ]

    # Find all heading positions
    headings = []
    for pattern in patterns:
        for m in re.finditer(pattern, full_text):
            heading = m.group(1).strip()
            if len(heading) > 3 and len(heading) < 80:
                headings.append((m.start(), heading))

    if len(headings) < 3:
        # Not enough headings detected, return as single section
        return [("Full Text", full_text)]

    headings.sort(key=lambda x: x[0])

    sections = []
    for i, (pos, heading) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(full_text)
        section_text = full_text[pos:end].strip()
        # Remove the heading from the text
        section_text = section_text[len(heading):].strip()
        if len(section_text.split()) > 20:  # skip tiny sections
            sections.append((heading, section_text))

    return sections if sections else [("Full Text", full_text)]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks of approximately chunk_size words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if len(chunk.split()) > 30:  # skip very small trailing chunks
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_paper(pages: list, paper_meta: dict) -> list:
    """
    Process extracted pages into chunks with metadata.
    Returns list of dicts with {id, text, metadata}.
    """
    # Combine all pages
    full_text = ' '.join(text for _, text in pages)

    # Try section-aware splitting
    sections = detect_sections(full_text)

    all_chunks = []
    chunk_idx = 0

    for section_name, section_text in sections:
        text_chunks = chunk_text(section_text)
        for chunk_text_str in text_chunks:
            chunk_id = f"{paper_meta['short_name']}_chunk_{chunk_idx:03d}"

            # Find which page this chunk is on (approximate)
            page_num = 1
            chunk_start = full_text.find(chunk_text_str[:50])
            if chunk_start >= 0:
                chars_before = chunk_start
                cumulative = 0
                for pn, pt in pages:
                    cumulative += len(pt)
                    if cumulative >= chars_before:
                        page_num = pn
                        break

            all_chunks.append({
                'id': chunk_id,
                'text': chunk_text_str,
                'metadata': {
                    'paper': paper_meta['short_name'],
                    'title': paper_meta['title'],
                    'section': section_name[:100],
                    'page': page_num,
                    'tags': paper_meta['tags'],
                }
            })
            chunk_idx += 1

    return all_chunks


# ==============================
# EMBEDDING + STORAGE
# ==============================

def build_index(force: bool = False):
    """Main entry point: extract PDFs → chunk → embed → store in ChromaDB."""

    # Check if vectorstore already exists
    if os.path.exists(VECTORSTORE_DIR) and not force:
        try:
            client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
            col = client.get_collection(COLLECTION_NAME)
            count = col.count()
            if count > 0:
                print(f"Vectorstore already exists with {count} chunks. Use --force to rebuild.")
                return
        except Exception:
            pass

    # Find PDFs
    if not os.path.exists(PAPERS_DIR):
        print(f"ERROR: Papers directory not found: {PAPERS_DIR}")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"ERROR: No PDFs found in {PAPERS_DIR}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDFs in {PAPERS_DIR}")

    # Extract and chunk all papers
    all_chunks = []
    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(PAPERS_DIR, pdf_file)
        meta = PDF_METADATA.get(pdf_file)
        if meta is None:
            # Auto-generate metadata for unknown PDFs
            short = pdf_file.replace('.pdf', '').replace(' ', '_')
            meta = {
                'short_name': short,
                'title': short,
                'tags': 'emg,activation'
            }

        print(f"\n  Processing: {pdf_file}")
        pages = extract_text_from_pdf(pdf_path)
        print(f"    Extracted {len(pages)} pages, {sum(len(t) for _, t in pages)} chars")

        chunks = chunk_paper(pages, meta)
        print(f"    Created {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute embeddings
    print("Computing embeddings...")
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"Computed {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # Store in ChromaDB
    print(f"\nStoring in ChromaDB at {VECTORSTORE_DIR}...")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Add in batches (ChromaDB limit is ~5000 per batch)
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[c['id'] for c in batch],
            documents=[c['text'] for c in batch],
            embeddings=[embeddings[i + j].tolist() for j in range(len(batch))],
            metadatas=[c['metadata'] for c in batch],
        )

    final_count = collection.count()
    print(f"\nDone! Stored {final_count} chunks in collection '{COLLECTION_NAME}'")

    # Print summary
    paper_counts = {}
    for c in all_chunks:
        p = c['metadata']['paper']
        paper_counts[p] = paper_counts.get(p, 0) + 1

    print("\nPer-paper breakdown:")
    for paper, count in sorted(paper_counts.items()):
        print(f"  {paper}: {count} chunks")


if __name__ == '__main__':
    force = '--force' in sys.argv
    build_index(force=force)
