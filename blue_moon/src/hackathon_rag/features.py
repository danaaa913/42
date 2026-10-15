"""Small opt-in interface for deterministic features discovered during the hackathon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from .models import Answer


@dataclass(frozen=True)
class FeaturePolicy:
    """Declares the minimum fields and retention behavior of one local feature."""

    feature_id: str
    accepted_fields: Tuple[str, ...] = ()
    transient_fields: Tuple[str, ...] = ()
    stores_user_values: bool = False

    def __post_init__(self) -> None:
        if not self.feature_id.strip():
            raise ValueError("feature_id cannot be empty")
        unknown = set(self.transient_fields) - set(self.accepted_fields)
        if unknown:
            raise ValueError("transient_fields must also be listed in accepted_fields")


@dataclass(frozen=True)
class FeatureContext:
    question: str
    language: str


class FeatureHandler(Protocol):
    policy: FeaturePolicy

    def can_handle(self, context: FeatureContext) -> bool: ...

    def handle(self, context: FeatureContext) -> Optional[Answer]: ...


class FeatureRegistry:
    """Runs explicitly registered local handlers before the document RAG path."""

    def __init__(self, handlers: Optional[List[FeatureHandler]] = None):
        self._handlers: List[FeatureHandler] = []
        for handler in handlers or []:
            self.register(handler)

    def register(self, handler: FeatureHandler) -> None:
        if any(item.policy.feature_id == handler.policy.feature_id for item in self._handlers):
            raise ValueError(f"Feature already registered: {handler.policy.feature_id}")
        self._handlers.append(handler)

    def policies(self) -> List[FeaturePolicy]:
        return [handler.policy for handler in self._handlers]

    def try_handle(self, question: str, language: str) -> Optional[Answer]:
        context = FeatureContext(question=question, language=language)
        for handler in self._handlers:
            if handler.can_handle(context):
                return handler.handle(context)
        return None
