"""Deterministic evidence fallback and document-quality warnings."""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

from .models import Answer, ParentSection, RetrievedChunk
from .retrieval import tokenize


AR_STOPWORDS = {"ما", "هي", "هو", "في", "من", "عن", "على", "الى", "إلى", "كيف", "هل", "شو", "ايش", "و", "او"}
EN_STOPWORDS = {"what", "which", "how", "is", "are", "the", "a", "an", "of", "for", "to", "in", "and", "or"}
SENTENCE_RE = re.compile(r"(?<=[.!?؟؛])\s+|\n+")
CLAIM_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>دينار(?:اً|ا)?|دنانير|ريال|جنيه|دولار|%|٪|"
    r"يوم(?:اً|ا)?|أيام|يوم عمل|ساعة|شهر|أشهر|سنة|سنوات|"
    r"days?|business days?|hours?|months?|years?|JOD|SAR|USD)",
    re.I,
)
OBSOLETE_RE = re.compile(r"\b(?:superseded|obsolete|expired|archived)\b|(?:ملغى|منتهي الصلاحية|نسخة قديمة|مؤرشف)", re.I)


def _query_terms(question: str) -> set[str]:
    return {token for token in tokenize(question) if token not in AR_STOPWORDS and token not in EN_STOPWORDS}


def _best_passages(question: str, retrieved: List[RetrievedChunk], parents: Dict[str, ParentSection]) -> List[Tuple[str, ParentSection]]:
    terms = _query_terms(question)
    candidates = []
    seen = set()
    for result in retrieved:
        parent = parents.get(result.chunk.parent_id)
        if parent is None or parent.id in seen:
            continue
        seen.add(parent.id)
        for sentence in SENTENCE_RE.split(parent.text):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            sentence_terms = set(tokenize(sentence))
            overlap = len(terms & sentence_terms)
            number_bonus = 1 if set(re.findall(r"\d+", question)) & set(re.findall(r"\d+", sentence)) else 0
            candidates.append((overlap + number_bonus, result.score, sentence, parent))
    candidates.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    if not candidates or candidates[0][0] < 1:
        return []
    output = []
    used_sources = set()
    for _, _, sentence, parent in candidates:
        key = (parent.source, parent.page, sentence)
        if key in used_sources:
            continue
        output.append((sentence[:800], parent))
        used_sources.add(key)
        if len(output) == 2:
            break
    return output


def extractive_fallback(
    question: str,
    retrieved: List[RetrievedChunk],
    parents: Dict[str, ParentSection],
    response_language: str,
    retrieved_summary: List[dict],
) -> Answer | None:
    """Return exact relevant passages when generation cannot be trusted."""
    passages = _best_passages(question, retrieved, parents)
    if not passages:
        return None
    arabic = response_language == "ar"
    intro = "النصوص الأكثر صلة في الوثائق:" if arabic else "Most relevant passages in the documents:"
    body = "\n\n".join(f"• {quote}" for quote, _ in passages)
    citations = []
    for index, (quote, parent) in enumerate(passages, 1):
        citations.append({
            "evidence_id": f"X{index}",
            "source": parent.source,
            "page": parent.page,
            "section": parent.section,
            "reason": "عرض حرفي آمن من المصدر" if arabic else "Safe verbatim extract from the source",
            "quote": quote,
            "evidence": quote,
        })
    return Answer(
        answer=f"{intro}\n\n{body}",
        supported=True,
        citations=citations,
        retrieved=retrieved_summary,
        answer_mode="extractive",
    )


def evidence_warnings(retrieved: List[RetrievedChunk], parents: Dict[str, ParentSection], language: str) -> List[str]:
    """Flag explicit obsolete markers and conflicting values with the same unit."""
    arabic = language == "ar"
    warnings = []
    values_by_unit = defaultdict(lambda: defaultdict(set))
    seen_parents = set()
    for result in retrieved:
        parent = parents.get(result.chunk.parent_id)
        if parent is None or parent.id in seen_parents:
            continue
        seen_parents.add(parent.id)
        if OBSOLETE_RE.search(parent.text) or OBSOLETE_RE.search(parent.source):
            warnings.append(
                f"المصدر {parent.source} يحتوي إشارة إلى أنه قديم أو ملغى؛ تحقّق من النسخة الرسمية الحالية."
                if arabic else
                f"Source {parent.source} is marked obsolete or archived; verify the current official version."
            )
        for match in CLAIM_RE.finditer(parent.text):
            unit = match.group("unit").casefold().replace("٪", "%")
            values_by_unit[unit][match.group("value")].add(parent.source)
    for unit, values in values_by_unit.items():
        sources = {source for source_set in values.values() for source in source_set}
        if len(values) > 1 and len(sources) > 1:
            display = "، ".join(sorted(values))
            warnings.append(
                f"وجدنا قيماً مختلفة للوحدة «{unit}» بين المصادر ({display})؛ قد تختلف حسب الحالة أو نسخة التعليمات."
                if arabic else
                f"Different values for “{unit}” appear across sources ({display}); they may reflect different cases or versions."
            )
    return list(dict.fromkeys(warnings))[:3]
