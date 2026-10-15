"""Environment-driven configuration for the local hackathon build."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv("hackathon.env")
except ImportError:
    pass


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HackathonConfig:
    documents_dir: Path = Path(os.getenv("HACKATHON_DOCUMENTS_DIR", "hackathon_workspace/documents"))
    index_dir: Path = Path(os.getenv("HACKATHON_INDEX_DIR", "hackathon_workspace/index"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    embedding_model: str = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    embedding_query_prefix: str = os.getenv("OLLAMA_EMBED_QUERY_PREFIX", "")
    embedding_document_prefix: str = os.getenv("OLLAMA_EMBED_DOCUMENT_PREFIX", "")
    collection_name: str = os.getenv("HACKATHON_COLLECTION", "hackathon_documents")
    parent_size: int = _int_env("HACKATHON_PARENT_SIZE", 3600)
    child_size: int = _int_env("HACKATHON_CHILD_SIZE", 900)
    child_overlap: int = _int_env("HACKATHON_CHILD_OVERLAP", 140)
    min_chunk_size: int = _int_env("HACKATHON_MIN_CHUNK_SIZE", 100)
    retrieve_candidates: int = _int_env("HACKATHON_RETRIEVE_CANDIDATES", 24)
    final_k: int = _int_env("HACKATHON_FINAL_K", 6)
    max_per_source: int = _int_env("HACKATHON_MAX_PER_SOURCE", 2)
    max_context_chars: int = _int_env("HACKATHON_MAX_CONTEXT_CHARS", 14000)
    request_timeout: int = _int_env("OLLAMA_REQUEST_TIMEOUT", 180)
    quarantine_suspicious: bool = _bool_env("HACKATHON_QUARANTINE_SUSPICIOUS", True)
    quarantine_sensitive: bool = _bool_env("HACKATHON_QUARANTINE_SENSITIVE", True)
    trace_enabled: bool = _bool_env("HACKATHON_TRACE_ENABLED", True)

    def validate(self) -> None:
        if self.child_size < 300:
            raise ValueError("HACKATHON_CHILD_SIZE must be at least 300")
        if not 0 <= self.child_overlap < self.child_size:
            raise ValueError("HACKATHON_CHILD_OVERLAP must be smaller than child size")
        if self.parent_size < self.child_size:
            raise ValueError("HACKATHON_PARENT_SIZE must be at least child size")
        if self.final_k < 1 or self.max_per_source < 1:
            raise ValueError("Retrieval limits must be positive")
