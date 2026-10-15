"""Context-preserving parent/child chunking for unknown competition documents."""

import hashlib
import re
from typing import Iterable, List, Sequence, Tuple

from .models import Chunk, ParentSection, SourceUnit


HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|(?:chapter|section|article)\s+\w+.*|"
    r"(?:الفصل|القسم|الباب|المادة)\s+[\w\u0600-\u06ff]+.*|"
    r"\d+(?:\.\d+)*[.)-]?\s+\S.+)$",
    re.I,
)
SERVICE_SECTION_RE = re.compile(
    r"^(?:[-*•]\s*)?(?:"
    r"الشروط|شروط الخدمة|شروط الاستحقاق|الفئات المستفيدة|المستفيدون|الوثائق المطلوبة|المستندات المطلوبة|"
    r"الرسوم|التكلفة|مدة الإنجاز|المدة|خطوات الخدمة|الإجراءات|طريقة التقديم|الاستثناءات|"
    r"قنوات الخدمة|معلومات الاتصال|الأساس القانوني|requirements?|eligibility|beneficiaries|"
    r"required documents?|fees?|cost|processing time|duration|steps?|procedure|exceptions?|"
    r"contact|legal basis)\s*[:：-]?\s*$",
    re.I,
)
COURSE_RE = re.compile(r"^(?:[A-Z]{2,6}\s*[- ]?\d{2,4})\s*[:\-–—]?")
SENTENCE_SEPARATORS = ["\n\n", "\n", "؟ ", ". ", "! ", "؛ ", "، ", " "]

SECTION_KINDS = {
    "eligibility": ("شروط الاستحقاق", "الفئات المستفيدة", "المستفيدون", "مؤهل", "eligibility", "beneficiaries"),
    "requirements": ("شروط الخدمة", "الشروط", "requirements", "requirement"),
    "documents": ("الوثائق المطلوبة", "المستندات المطلوبة", "required document", "documents"),
    "fees": ("الرسوم", "التكلفة", "دينار", "fees", "cost"),
    "duration": ("مدة الإنجاز", "المدة", "يوم عمل", "processing time", "duration"),
    "steps": ("خطوات الخدمة", "الإجراءات", "طريقة التقديم", "steps", "procedure", "apply"),
    "location": ("مكان تقديم الخدمة", "أماكن تقديم الخدمة", "قنوات تقديم الخدمة", "service location", "locations"),
    "exceptions": ("الاستثناءات", "exceptions"),
    "contact": ("قنوات الخدمة", "معلومات الاتصال", "contact"),
    "legal_basis": ("الأساس القانوني", "المادة", "النظام", "legal basis", "article"),
}


def classify_section(section: str, text: str = "") -> str:
    """Classify common government-service sections without using an LLM."""
    # The explicit heading is stronger evidence than words mentioned inside the
    # section. For example, procedural steps often mention fees or documents.
    heading = section.casefold()
    for kind, markers in SECTION_KINDS.items():
        if any(marker.casefold() in heading for marker in markers):
            return kind
    value = text[:240].casefold()
    for kind, markers in SECTION_KINDS.items():
        if any(marker.casefold() in value for marker in markers):
            return kind
    return "general"


def _stable_id(*parts: object) -> str:
    value = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _overlap_tail(text: str, overlap: int) -> str:
    if not overlap or len(text) <= overlap:
        return text if overlap else ""
    tail = text[-overlap:]
    boundary = re.search(r"[\s،؛,.!?؟]", tail)
    return tail[boundary.end():].lstrip() if boundary else tail


def _join_with_limit(parts: Sequence[str], limit: int) -> List[str]:
    output, current = [], ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{current}\n\n{part}".strip() if current else part
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                output.append(current)
            current = part
    if current:
        output.append(current)
    return output


def _split_long(text: str, limit: int, overlap: int) -> List[str]:
    if len(text) <= limit:
        return [text.strip()]
    for separator in SENTENCE_SEPARATORS:
        if separator not in text:
            continue
        pieces = text.split(separator)
        chunks, current = [], ""
        for index, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            suffix = separator.strip() if index < len(pieces) - 1 else ""
            candidate_piece = f"{piece}{suffix}"
            candidate = f"{current} {candidate_piece}".strip() if current else candidate_piece
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
                tail = _overlap_tail(current, overlap)
                current = f"{tail} {candidate_piece}".strip()
            else:
                break
        if current:
            chunks.append(current)
        if chunks and all(len(chunk) <= limit for chunk in chunks):
            return chunks
    step = max(1, limit - overlap)
    return [text[start:start + limit].strip() for start in range(0, len(text), step) if text[start:start + limit].strip()]


def _split_table(text: str, limit: int) -> List[str]:
    """Split Markdown tables by complete rows and repeat the column header."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prefix = []
    if lines and lines[0].casefold() == "[table]":
        prefix.append(lines.pop(0))
    if len(lines) < 2 or not lines[0].startswith("|") or "---" not in lines[1]:
        return []
    header = prefix + lines[:2]
    rows = lines[2:]
    if not rows:
        return ["\n".join(header)]
    base = "\n".join(header)
    chunks = []
    current_rows: List[str] = []
    for row in rows:
        candidate = "\n".join(header + current_rows + [row])
        if current_rows and len(candidate) > limit:
            chunks.append("\n".join(header + current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)
    if current_rows:
        chunks.append("\n".join(header + current_rows))
    return chunks or [base]


def _structural_blocks(text: str) -> List[Tuple[str, str]]:
    """Return (section title, text) blocks while retaining headings/course labels."""
    blocks = []
    section = ""
    current = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current and current[-1] != "":
                current.append("")
            continue
        is_boundary = bool(HEADING_RE.match(stripped) or COURSE_RE.match(stripped) or SERVICE_SECTION_RE.match(stripped))
        # Numbered service steps and requirements are list items, not section
        # headings. Keeping the complete list together preserves the procedure's
        # meaning and prevents tiny chunks such as a single isolated step.
        if re.match(r"^\d+[.)-]\s+", stripped) and classify_section(section) != "general":
            is_boundary = False
        if is_boundary and current:
            blocks.append((section, "\n".join(current).strip()))
            current = []
        if is_boundary:
            section = stripped.lstrip("# ").strip()
        current.append(stripped)
    if current:
        blocks.append((section, "\n".join(current).strip()))
    if len(blocks) > 1:
        first_section, first_body = blocks[0]
        first_lines = [line.strip().lstrip("# ") for line in first_body.splitlines() if line.strip()]
        if first_section and len(first_lines) == 1 and first_lines[0] == first_section:
            document_title = first_section
            blocks = blocks[1:]
            blocks = [
                (f"{document_title} > {block_section}" if block_section else document_title, body)
                for block_section, body in blocks
            ]
    return blocks or [("", text.strip())]


def _merge_small(chunks: List[str], minimum: int, maximum: int) -> List[str]:
    if len(chunks) < 2:
        return chunks
    merged = []
    for chunk in chunks:
        if len(chunk) < minimum and merged and len(merged[-1]) + len(chunk) + 2 <= maximum:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)
    if len(merged) > 1 and len(merged[0]) < minimum and len(merged[0]) + len(merged[1]) + 2 <= maximum:
        merged[1] = f"{merged[0]}\n\n{merged[1]}"
        merged.pop(0)
    return merged


def _context_header(source: str, page: int | None, section: str, document_type: str, section_kind: str = "general") -> str:
    fields = [f"Source: {source}"]
    if page is not None:
        fields.append(f"Page: {page}")
    if section:
        fields.append(f"Section: {section}")
    fields.append(f"Type: {document_type}")
    if section_kind != "general":
        fields.append(f"Service section: {section_kind}")
    return "[" + " | ".join(fields) + "]"


def build_chunks(
    units: Iterable[SourceUnit],
    parent_size: int = 3600,
    child_size: int = 900,
    overlap: int = 140,
    minimum: int = 100,
) -> Tuple[List[ParentSection], List[Chunk]]:
    """Build searchable child chunks and larger answer-context parents."""
    parents: List[ParentSection] = []
    children: List[Chunk] = []
    for unit in units:
        for structural_section, block in _structural_blocks(unit.text):
            section = structural_section or unit.section
            section_kind = classify_section(section, block)
            is_table = unit.document_type in {"xlsx", "docx_table"} or block.lstrip().casefold().startswith("[table]")
            parent_bodies = _split_table(block, parent_size) if is_table else []
            if not parent_bodies:
                parent_bodies = _split_long(block, parent_size, min(overlap * 2, parent_size // 4))
            for parent_index, parent_body in enumerate(parent_bodies):
                parent_id = _stable_id(unit.source, unit.page, section, parent_index, parent_body)
                parent = ParentSection(
                    id=parent_id,
                    text=parent_body,
                    source=unit.source,
                    page=unit.page,
                    section=section,
                    document_type=unit.document_type,
                    section_kind=section_kind,
                )
                parents.append(parent)
                bodies = _split_table(parent_body, child_size) if is_table else []
                if not bodies:
                    bodies = _merge_small(_split_long(parent_body, child_size, overlap), minimum, child_size)
                for child_index, body in enumerate(bodies):
                    header = _context_header(unit.source, unit.page, section, unit.document_type, section_kind)
                    text = f"{header}\n{body}"
                    children.append(Chunk(
                        id=_stable_id(parent_id, child_index, body),
                        parent_id=parent_id,
                        text=text,
                        body=body,
                        source=unit.source,
                        page=unit.page,
                        section=section,
                        document_type=unit.document_type,
                        chunk_index=child_index,
                        section_kind=section_kind,
                    ))
    return parents, children
