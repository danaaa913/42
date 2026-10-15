"""Local, document-grounded RAG toolkit prepared for the AI hackathon."""

from .config import HackathonConfig
from .pipeline import HackathonRAG

__all__ = ["HackathonConfig", "HackathonRAG"]
