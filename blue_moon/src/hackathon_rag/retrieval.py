"""Hybrid vector + BM25 retrieval with RRF and source diversity."""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from .models import Chunk, RetrievedChunk
from .ollama_client import OllamaClient
from .store import DocumentStore


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u0600-\u06ff]+")

QUERY_SECTION_MARKERS = {
    "eligibility": ("مؤهل", "استحقاق", "الفئة", "eligible", "eligibility"),
    "requirements": ("الشروط", "شرط", "requirements", "conditions"),
    "documents": ("وثائق", "مستندات", "اوراق", "أوراق", "documents", "papers"),
    "fees": ("رسوم", "تكلفة", "كم السعر", "fees", "cost", "price"),
    "duration": ("مدة", "كم يوم", "متى", "duration", "how long", "processing time"),
    "steps": ("خطوات", "كيف اقدم", "كيف أقدم", "اجراءات", "إجراءات", "steps", "how to apply"),
    "location": ("مكان", "أماكن", "وين", "أين", "موقع", "فرع", "location", "where"),
    "exceptions": ("استثناء", "exceptions"),
    "contact": ("اتصال", "هاتف", "ايميل", "بريد", "contact", "phone", "email"),
    "legal_basis": ("المادة", "قانون", "نظام", "legal basis", "article", "law"),
}


def query_section_kinds(question: str) -> set[str]:
    value = question.casefold()
    return {
        kind for kind, markers in QUERY_SECTION_MARKERS.items()
        if any(marker.casefold() in value for marker in markers)
    }


def tokenize(text: str) -> List[str]:
    text = ARABIC_DIACRITICS.sub("", text.casefold())
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    tokens = [token.strip("،؛؟ـ") for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if token and (len(token) > 1 or token.isdigit())]


class LightweightBM25:
    """Dependency-free BM25 suitable for a hackathon-sized corpus."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.tokens = [tokenize(document) for document in documents]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        document_frequency = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(self.tokens)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> List[float]:
        terms = tokenize(query)
        output = []
        for tokens in self.tokens:
            frequencies = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / max(1.0, self.avgdl))
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            output.append(score)
        return output


class HybridRetriever:
    def __init__(
        self,
        store: DocumentStore,
        ollama: OllamaClient,
        embedding_model: str,
        query_prefix: str = "",
        candidates: int = 24,
        final_k: int = 6,
        max_per_source: int = 2,
    ):
        self.store = store
        self.ollama = ollama
        self.embedding_model = embedding_model
        self.query_prefix = query_prefix
        self.candidates = candidates
        self.final_k = final_k
        self.max_per_source = max_per_source
        self.chunks = store.load_chunks()
        self.by_id = {chunk.id: chunk for chunk in self.chunks}
        self.bm25 = LightweightBM25([chunk.text for chunk in self.chunks])

    def retrieve(self, question: str, source_scope: List[str] | None = None) -> List[RetrievedChunk]:
        allowed_sources = set(source_scope or [])
        scoped_chunks = [chunk for chunk in self.chunks if not allowed_sources or chunk.source in allowed_sources]
        if not scoped_chunks:
            return []
        query_vector = self.ollama.embed(self.embedding_model, [self.query_prefix + question], batch_size=1)[0]
        try:
            vector_ids = self.store.vector_ids(
                query_vector,
                min(self.candidates, len(scoped_chunks)),
                sorted(allowed_sources) if allowed_sources else None,
            )
        except TypeError:  # compatibility for small test stores and external adapters
            vector_ids = self.store.vector_ids(query_vector, min(self.candidates, len(self.chunks)))
            vector_ids = [chunk_id for chunk_id in vector_ids if not allowed_sources or self.by_id[chunk_id].source in allowed_sources]
        vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids, 1)}

        keyword_scores = self.bm25.scores(question)
        keyword_indices = sorted(range(len(keyword_scores)), key=keyword_scores.__getitem__, reverse=True)
        keyword_indices = [
            index for index in keyword_indices
            if keyword_scores[index] > 0 and (not allowed_sources or self.chunks[index].source in allowed_sources)
        ][:self.candidates]
        keyword_rank = {self.chunks[index].id: rank for rank, index in enumerate(keyword_indices, 1)}

        fused: Dict[str, float] = defaultdict(float)
        for chunk_id, rank in vector_rank.items():
            fused[chunk_id] += 1.0 / (60 + rank)
        for chunk_id, rank in keyword_rank.items():
            fused[chunk_id] += 1.0 / (60 + rank)

        query_numbers = set(re.findall(r"\d+", question))
        desired_sections = query_section_kinds(question)
        for chunk_id in list(fused):
            if query_numbers and query_numbers & set(re.findall(r"\d+", self.by_id[chunk_id].text)):
                fused[chunk_id] *= 1.15
            if desired_sections and self.by_id[chunk_id].section_kind in desired_sections:
                fused[chunk_id] *= 1.18

        ranked = sorted(fused, key=fused.get, reverse=True)
        selected = []
        source_counts = Counter()
        seen_parents = set()

        def add_result(chunk_id: str) -> None:
            chunk = self.by_id[chunk_id]
            selected.append(RetrievedChunk(
                chunk=chunk,
                score=fused[chunk_id],
                vector_rank=vector_rank.get(chunk_id),
                keyword_rank=keyword_rank.get(chunk_id),
            ))
            source_counts[chunk.source] += 1
            seen_parents.add(chunk.parent_id)

        # Prefer source diversity with a soft penalty. A hard per-source cap can
        # discard multi-page evidence when the answer lives in one long document.
        remaining = list(ranked)
        while remaining and len(selected) < self.final_k:
            eligible = [chunk_id for chunk_id in remaining if self.by_id[chunk_id].parent_id not in seen_parents]
            if not eligible:
                break

            def adjusted_score(chunk_id: str) -> float:
                source = self.by_id[chunk_id].source
                excess = max(0, source_counts[source] - self.max_per_source + 1)
                return fused[chunk_id] * (0.72 ** excess)

            chosen = max(eligible, key=lambda chunk_id: (adjusted_score(chunk_id), fused[chunk_id]))
            add_result(chosen)
            remaining.remove(chosen)
        return selected
