"""Small serializable models shared by the hackathon RAG modules."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SourceUnit:
    text: str
    source: str
    page: Optional[int] = None
    section: str = ""
    document_type: str = "text"
    section_kind: str = "general"


@dataclass
class ParentSection:
    id: str
    text: str
    source: str
    page: Optional[int] = None
    section: str = ""
    document_type: str = "text"
    section_kind: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
@dataclass
class Chunk:
    id: str
    parent_id: str
    text: str
    body: str
    source: str
    page: Optional[int] = None
    section: str = ""
    document_type: str = "text"
    chunk_index: int = 0
    section_kind: str = "general"

    def metadata(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "source": self.source,
            "page": self.page if self.page is not None else -1,
            "section": self.section,
            "document_type": self.document_type,
            "chunk_index": self.chunk_index,
            "section_kind": self.section_kind,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None


@dataclass
class Answer:
    answer: str
    supported: bool
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    timings_ms: Dict[str, int] = field(default_factory=dict)
    workflow: List[str] = field(default_factory=list)
    request_id: str = ""
    privacy_action: str = "allow"
    privacy_categories: List[str] = field(default_factory=list)
    redacted_count: int = 0
    privacy_notice: str = ""
    answer_mode: str = "generated"
    evidence_warnings: List[str] = field(default_factory=list)
    scope_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
