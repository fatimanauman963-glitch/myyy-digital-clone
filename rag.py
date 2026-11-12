"""Retrieval part of the digital clone: chunking, BM25, dense search, RRF fusion."""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMB_PATH = BASE_DIR / "embeddings.npy"
CHUNKS_PATH = BASE_DIR / "chunks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 250   # about 200-300 words per chunk
OVERLAP_WORDS = 40  # small overlap between neighbouring chunks
RRF_K = 60          # Reciprocal Rank Fusion constant: score = sum 1 / (60 + rank)

MODE_BM25 = "BM25 only"
MODE_DENSE = "Dense only"
MODE_HYBRID = "Hybrid (RRF)"
MODES = [MODE_BM25, MODE_DENSE, MODE_HYBRID]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "to", "of", "in", "on",
    "at", "for", "and", "or", "but", "with", "by", "it", "this", "that", "do", "does",
    "did", "you", "your", "i", "my", "me", "what", "which", "who", "how", "from", "as",
    "so", "if", "can", "about",
}


# ---------------------------------------------------------------- chunking
def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def build_chunks():
    """Read every .md / .txt file in data/ and return chunks with metadata."""
    chunks = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for n, piece in enumerate(chunk_text(text)):
            chunks.append({
                "source": path.name,
                "chunk_id": f"{path.stem}-{n}",
                "text": piece,
            })
    return chunks


# --------------------------------------------------------------- retriever
class Retriever:
    def __init__(self, rebuild=False):
        self.chunks = build_chunks()
        if not self.chunks:
            raise RuntimeError("No .md/.txt files found in data/")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._model = None
        self.embeddings = self._load_or_build_embeddings(rebuild)

    # -- dense side
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def _load_or_build_embeddings(self, rebuild):
        texts = [c["text"] for c in self.chunks]
        if not rebuild and EMB_PATH.exists() and CHUNKS_PATH.exists():
            saved = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
            emb = np.load(EMB_PATH)
            if saved == texts and emb.shape[0] == len(texts):
                return emb
        emb = self._get_model().encode(texts, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        try:
            np.save(EMB_PATH, emb)
            CHUNKS_PATH.write_text(json.dumps(texts), encoding="utf-8")
        except OSError:
            pass  # read-only file system is fine, we just recompute next start
        return emb

    # -- single retrievers: return list of (chunk_index, score)
    def bm25_search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def dense_search(self, query, k=5):
        q = self._get_model().encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q  # cosine similarity (vectors are normalised)
        order = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in order]

    # -- fusion
    def hybrid_search(self, query, k=5, pool=20):
        bm = self.bm25_search(query, pool)
        de = self.dense_search(query, pool)
        bm_rank = {i: r for r, (i, _) in enumerate(bm, start=1)}
        de_rank = {i: r for r, (i, _) in enumerate(de, start=1)}
        fused = {}
        for ranks in (bm_rank, de_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{
            "chunk": self.chunks[i],
            "score": fused[i],
            "bm25_rank": bm_rank.get(i),
            "dense_rank": de_rank.get(i),
        } for i in order]

    # -- one entry point used by the app
    def search(self, query, mode=MODE_HYBRID, k=4):
        if mode == MODE_HYBRID:
            return self.hybrid_search(query, k)
        pairs = self.bm25_search(query, k) if mode == MODE_BM25 else self.dense_search(query, k)
        return [{
            "chunk": self.chunks[i], "score": s,
            "bm25_rank": None, "dense_rank": None,
        } for i, s in pairs]
