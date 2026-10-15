"""Conservative cleanup that preserves document structure and tables."""

import re
import unicodedata
from collections import Counter
from typing import Iterable, List

from .models import SourceUnit


INVISIBLE_CHARS = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
BIDI_CONTROLS = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
BOILERPLATE_LINES = [
    re.compile(r"^you need to enable javascript", re.I),
    re.compile(r"^يجب عليك تفعيل (?:الجافاسكربت|javascript)", re.I),
    re.compile(r"^(?:التالي|السابق|البداية|النهاية)(?:\s+(?:التالي|السابق|البداية|النهاية))*$"),
    re.compile(r"^(?:all rights reserved|جميع الحقوق محفوظة)", re.I),
]


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = INVISIBLE_CHARS.sub("", line)
    line = BIDI_CONTROLS.sub("", line)
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def clean_text(text: str) -> str:
    """Remove extraction noise without flattening paragraph/table boundaries."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[A-Za-z])", "", text)
    lines = []
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if any(pattern.search(line) for pattern in BOILERPLATE_LINES):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def remove_repeated_page_lines(units: Iterable[SourceUnit]) -> List[SourceUnit]:
    """Remove short headers/footers repeated on many pages of the same file."""
    grouped = {}
    for unit in units:
        grouped.setdefault(unit.source, []).append(unit)

    result = []
    for source_units in grouped.values():
        page_units = [unit for unit in source_units if unit.page is not None]
        repeated = set()
        if len(page_units) >= 3:
            counts = Counter()
            for unit in page_units:
                page_lines = [line for line in unit.text.splitlines() if normalize_line(line)]
                edge_lines = page_lines[:2] + page_lines[-2:]
                seen_on_page = {
                    normalize_line(line).casefold()
                    for line in edge_lines
                    if 2 <= len(normalize_line(line)) <= 120
                }
                counts.update(seen_on_page)
            threshold = max(3, (len(page_units) + 2) // 3)
            repeated = {line for line, count in counts.items() if count >= threshold}

        for unit in source_units:
            lines = [
                line for line in unit.text.splitlines()
                if normalize_line(line).casefold() not in repeated
            ]
            result.append(SourceUnit(
                text=clean_text("\n".join(lines)),
                source=unit.source,
                page=unit.page,
                section=unit.section,
                document_type=unit.document_type,
            ))
    return [unit for unit in result if unit.text]
