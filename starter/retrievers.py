"""Retrieval implementations for Day 7: Lexical (TF-IDF), Dense, and Hybrid (RRF)."""
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from baseline import tokenize, pack_context, hit

class TfidfRetriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if filters:
                meta = chunk.get('metadata', {})
                if any(meta.get(k) != v for k, v in filters.items()):
                    continue
            score = sum(val * vector.get(term, 0.0) for term, val in q_vec.items())
            if score > 0:
                results.append({**chunk, 'score': float(score), 'strategy': 'tfidf'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class DenseRetriever:
    def __init__(self, chunks: list[dict[str, Any]], embeddings_path: Path, embedder=None):
        self.chunks = {c['id']: c for c in chunks}
        self.chunk_list = chunks
        raw_embs = json.loads(embeddings_path.read_text(encoding='utf-8'))
        self.embeddings = {item['id']: item['embedding'] for item in raw_embs}
        self.embedder = embedder

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1)) or 1.0
        norm2 = math.sqrt(sum(b * b for b in v2)) or 1.0
        return dot / (norm1 * norm2)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        q_vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        
        candidates = []
        for c in self.chunk_list:
            if filters:
                meta = c.get('metadata', {})
                if any(meta.get(k) != v for k, v in filters.items()):
                    continue
            c_vec = self.embeddings.get(c['id'])
            if not c_vec:
                continue
            sim = self._cosine_similarity(q_vec, c_vec)
            candidates.append({**c, 'score': float(sim), 'strategy': 'dense'})
            
        return sorted(candidates, key=lambda x: (-x['score'], x['id']))[:top_k]


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) between Dense and TF-IDF."""
    def __init__(self, tfidf_retriever: TfidfRetriever, dense_retriever: DenseRetriever, rrf_k: int = 60):
        self.tfidf = tfidf_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        # Retrieve candidate pool of top 10 from each strategy
        pool_k = max(top_k * 3, 10)
        lex_results = self.tfidf.search(query, filters=filters, top_k=pool_k)
        dense_results = self.dense.search(query, filters=filters, top_k=pool_k)

        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict[str, Any]] = {}

        for rank, c in enumerate(lex_results, start=1):
            cid = c['id']
            chunks_by_id[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, c in enumerate(dense_results, start=1):
            cid = c['id']
            chunks_by_id[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        merged = []
        for cid, score in rrf_scores.items():
            c = dict(chunks_by_id[cid])
            c['score'] = score
            c['strategy'] = 'hybrid_rrf'
            merged.append(c)

        return sorted(merged, key=lambda x: (-x['score'], x['id']))[:top_k]

