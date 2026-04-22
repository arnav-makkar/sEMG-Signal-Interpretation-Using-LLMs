#!/usr/bin/env python3
"""
RAG Retriever — ChromaDB Semantic Search over Research Papers
==============================================================
Primary: Queries ChromaDB vectorstore built from real paper PDFs.
Fallback: TF-IDF over knowledge_base.json if vectorstore not available.

Usage:
    from rag.retriever import EMGPaperRetriever
    retriever = EMGPaperRetriever()
    chunks = retriever.retrieve("muscle activation onset detection", top_k=5)
    context = retriever.format_context(chunks)
"""

import json
import math
import os
import re
import warnings
from collections import Counter
from typing import List, Dict, Optional

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), 'vectorstore')
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
COLLECTION_NAME = 'emg_papers'


class EMGPaperRetriever:
    """
    Semantic retriever over EMG research paper knowledge.
    Uses ChromaDB (sentence-transformers embeddings) when available,
    falls back to TF-IDF over knowledge_base.json.
    """

    def __init__(self, vectorstore_path: str = VECTORSTORE_DIR,
                 fallback_kb_path: str = KNOWLEDGE_BASE_PATH):
        self.use_chromadb = False
        self.collection = None

        # Try ChromaDB first
        try:
            import chromadb
            if os.path.exists(vectorstore_path):
                client = chromadb.PersistentClient(path=vectorstore_path)
                self.collection = client.get_collection(COLLECTION_NAME)
                count = self.collection.count()
                if count > 0:
                    self.use_chromadb = True
                    self._chunk_count = count
        except Exception:
            pass

        # Fallback to TF-IDF
        if not self.use_chromadb:
            self._load_tfidf_fallback(fallback_kb_path)

    # ------------------------------------------------------------------
    # TF-IDF fallback (from knowledge_base.json)
    # ------------------------------------------------------------------

    def _load_tfidf_fallback(self, kb_path: str):
        if not os.path.exists(kb_path):
            self._fallback_chunks = []
            return
        with open(kb_path) as f:
            data = json.load(f)
        self._fallback_chunks = data.get('chunks', [])
        self._build_tfidf_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-z0-9]+', text.lower())

    def _build_tfidf_index(self):
        N = len(self._fallback_chunks)
        self._tf = []
        df = Counter()
        for chunk in self._fallback_chunks:
            combined = chunk['text'] + ' ' + ' '.join(chunk.get('tags', []) * 3)
            tokens = self._tokenize(combined)
            tf = Counter(tokens)
            total = len(tokens) or 1
            self._tf.append({t: c / total for t, c in tf.items()})
            for term in set(tokens):
                df[term] += 1
        self._idf = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}

    def _tfidf_retrieve(self, query: str, top_k: int) -> List[Dict]:
        tokens = self._tokenize(query)
        q_tf = Counter(tokens)
        total = len(tokens) or 1
        q_vec = {t: (c / total) * self._idf.get(t, 1.0) for t, c in q_tf.items()}

        scores = []
        for i, tf in enumerate(self._tf):
            doc_vec = {t: v * self._idf.get(t, 1.0) for t, v in tf.items()}
            common = set(q_vec) & set(doc_vec)
            dot = sum(q_vec[t] * doc_vec[t] for t in common)
            mag_a = math.sqrt(sum(v ** 2 for v in q_vec.values()))
            mag_b = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
            score = dot / (mag_a * mag_b) if mag_a > 1e-9 and mag_b > 1e-9 else 0
            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for _, i in scores[:top_k]:
            c = self._fallback_chunks[i]
            results.append({
                'id': c.get('id', f'chunk_{i}'),
                'text': c['text'],
                'paper': c.get('paper', 'unknown'),
                'metadata': c,
            })
        return results

    # ------------------------------------------------------------------
    # ChromaDB retrieval
    # ------------------------------------------------------------------

    def _chroma_retrieve(self, query: str, top_k: int,
                         tag_filter: Optional[List[str]] = None) -> List[Dict]:
        kwargs = {
            'query_texts': [query],
            'n_results': top_k,
            'include': ['documents', 'metadatas', 'distances'],
        }

        # Optional tag filtering
        if tag_filter and len(tag_filter) == 1:
            kwargs['where'] = {"tags": {"$contains": tag_filter[0]}}
        elif tag_filter and len(tag_filter) > 1:
            kwargs['where'] = {
                "$or": [{"tags": {"$contains": t}} for t in tag_filter]
            }

        try:
            results = self.collection.query(**kwargs)
        except Exception:
            # If filtering fails, retry without filter
            kwargs.pop('where', None)
            results = self.collection.query(**kwargs)

        chunks = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                chunks.append({
                    'id': doc_id,
                    'text': results['documents'][0][i],
                    'paper': meta.get('paper', 'unknown'),
                    'title': meta.get('title', ''),
                    'section': meta.get('section', ''),
                    'page': meta.get('page', 0),
                    'metadata': meta,
                })
        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5,
                 tag_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieve top_k most relevant paper chunks for a text query.
        Uses ChromaDB semantic search or TF-IDF fallback.
        """
        if self.use_chromadb:
            return self._chroma_retrieve(query, top_k, tag_filter)
        else:
            return self._tfidf_retrieve(query, top_k)

    def retrieve_for_workflow(self, workflow: str,
                              signal_flags: Optional[List[str]] = None,
                              top_k: int = 5) -> List[Dict]:
        """
        Workflow-aware retrieval. Builds a rich query from the workflow name
        and detected signal flags, then retrieves relevant paper chunks.

        workflow: 'activation' | 'fatigue' | 'quality' | 'drift'
        signal_flags: list of detected signal conditions to expand the query
        """
        base_queries = {
            'activation': (
                "muscle activation onset offset duty cycle spatial distribution "
                "sensor array gesture pattern dominant muscle temporal profile consistency "
                "motor unit recruitment activation level co-contraction"
            ),
            'fatigue': (
                "muscle fatigue MDF MNF median frequency spectral compression "
                "conduction velocity motor unit synchronization entropy RMS increase "
                "fatigue index onset detection"
            ),
            'quality': (
                "signal quality SNR baseline noise powerline artifact clipping "
                "crosstalk dropout electrode contact spectral entropy kurtosis"
            ),
            'drift': (
                "baseline drift electrode impedance perspiration DC offset "
                "motion artifact warm-up stabilization spatial reconfiguration "
                "amplitude frequency dissociation"
            ),
        }

        flag_terms = {
            'motor_unit_recruitment': 'WAMP ZC fast-twitch recruitment Henneman size principle',
            'spatial_compensation':   'activation map cosine similarity spatial pattern redistribution',
            'co_contraction':         'agonist antagonist co-contraction simultaneous activation',
            'rms_variability':        'consistency CV coefficient of variation signal stability',
            'normalization':          'MVC normalization amplitude comparison cross-subject',
            'onset_detection':        'onset offset threshold consecutive windows burst duration',
            'duty_cycle':             'duty cycle sustained intermittent activation percentage',
            'mdf_decline':            'MDF decline fatigue spectral compression frequency shift',
            'lh_ratio_increase':      'low high frequency ratio spectral compression power band',
            'entropy_decrease':       'sample entropy synchronization complexity regularity',
            'high_kurtosis':          'kurtosis spike artifact distribution tail',
            'powerline_detected':     'powerline 50 60 Hz interference notch filter',
            'crosstalk_detected':     'crosstalk electrode spacing correlation adjacent muscles',
            'motion_artifact':        'motion artifact low frequency DC offset skin stretch',
            'baseline_drift':         'baseline drift linear detrending correction wander',
            'high_duty_cycle':        'sustained contraction continuous activation endurance',
            'frequency_shift':        'frequency shift recruitment motor unit spectral change',
        }

        query_parts = [base_queries.get(workflow, workflow)]
        for flag in (signal_flags or []):
            if flag in flag_terms:
                query_parts.append(flag_terms[flag])

        query = ' '.join(query_parts)
        return self.retrieve(query, top_k=top_k)

    def format_context(self, chunks: List[Dict], header: str = "RESEARCH_CONTEXT") -> str:
        """Format retrieved chunks as a compact block for LLM prompt injection."""
        lines = [f"<{header}>"]
        seen_texts = set()
        for c in chunks:
            # Deduplicate near-identical chunks
            text_key = c['text'][:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            paper = c.get('title') or c.get('paper', 'unknown')
            section = c.get('section', '')
            source_line = f"[{paper}"
            if section:
                source_line += f" — {section}"
            source_line += "]"

            lines.append(source_line)
            lines.append(c['text'][:800])  # cap chunk length in prompt
            lines.append("")
        lines.append(f"</{header}>")
        return "\n".join(lines)

    def list_papers(self) -> List[str]:
        """List unique papers in the knowledge base."""
        if self.use_chromadb:
            results = self.collection.get(include=['metadatas'])
            papers = set()
            for m in results['metadatas']:
                papers.add(m.get('title', m.get('paper', 'unknown')))
            return sorted(papers)
        else:
            seen = []
            for c in self._fallback_chunks:
                if c.get('paper') not in seen:
                    seen.append(c['paper'])
            return seen

    def get_backend(self) -> str:
        """Return which backend is active."""
        if self.use_chromadb:
            return f"ChromaDB ({self._chunk_count} chunks)"
        else:
            return f"TF-IDF fallback ({len(self._fallback_chunks)} chunks)"


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retriever = EMGPaperRetriever()

    print(f"Backend: {retriever.get_backend()}")
    print(f"Papers: {retriever.list_papers()}\n")

    queries = [
        ("activation", "muscle activation spatial pattern onset detection duty cycle"),
        ("fatigue",    "MDF spectral compression fatigue motor unit synchronization"),
        ("quality",    "baseline noise SNR powerline artifact crosstalk"),
    ]

    for label, q in queries:
        print(f"=== Query: {label} ===")
        chunks = retriever.retrieve(q, top_k=3)
        for c in chunks:
            print(f"  [{c['paper']}] {c['text'][:80]}...")
        print()
