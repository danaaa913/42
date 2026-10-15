"""Shared answer engine for the interactive UI and the competition delivery path.

The tested competition logic lives in the repository-root ``main.py`` (routing,
``retrieve_contexts``, ``missing_fee_details``, prompt building, the Ollama call
and the fee-coverage repair).  This module is a *thin adapter* over that code:
it imports ``main`` and reuses its functions directly, exactly like
``hackathon_workspace/eval_results/retrieval_audit.py`` already does.

Nothing about routing, retrieval, prompts, fee detection or the answer/contexts
format is re-implemented here.  The only additions are the things an interactive
chat needs and the batch runner does not:

* per-session conversation memory (history, previous service and topics),
* a single-concurrency lock so the local model is never called in parallel,
* lazily loading the persisted index once and reusing it for every question.

There is therefore no second implementation to keep in sync.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as competition  # noqa: E402  (the single competition engine)

# Keep a small, bounded, in-memory history per session.  Nothing is written to
# disk and full questions are never logged.
MAX_HISTORY_TURNS = 12

SENSITIVE_REPLY = (
    "لحماية خصوصيتك، لا تشارك رموز التحقق أو كلمات المرور. "
    "أعد السؤال من دون البيانات السرية."
)
SENSITIVE_HISTORY_PLACEHOLDER = "[رسالة تحتوي بيانات سرية محجوبة]"
MODEL_UNAVAILABLE_REPLY = "تعذر توليد الإجابة الآن؛ يرجى إعادة المحاولة."
ENGINE_UNAVAILABLE_AR = (
    "تعذر الاتصال بمحرك Ollama المحلي. تأكد من تشغيل Ollama ومن توفر النموذج "
    "المطلوب ثم أعد المحاولة."
)


class EngineUnavailable(RuntimeError):
    """Raised when the local Ollama engine cannot be reached."""


class CompetitionEngine:
    """Stateful, single-concurrency wrapper around the competition engine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._services: dict[str, dict] | None = None
        self._retriever: Any | None = None
        self._sessions: dict[str, dict] = {}

    # ------------------------------------------------------------------ setup
    def _ensure_ready(self) -> tuple[dict[str, dict], Any]:
        """Load (or build once) the persisted index; reuse it afterwards."""
        if self._retriever is None:
            services = competition.parse_services()
            retriever = competition.ensure_index(services)
            self._services = services
            self._retriever = retriever
        return self._services, self._retriever

    # ----------------------------------------------------------------- health
    def health(self) -> dict:
        """Report index and Ollama readiness without building anything."""
        manifest = competition.INDEX_DIR / "manifest.json"
        index_ready = manifest.exists()
        models: list[str] = []
        ollama_ready = False
        try:
            with request.urlopen(competition.OLLAMA + "/api/tags", timeout=3) as response:
                payload = json.load(response)
            models = [item.get("name", "") for item in payload.get("models", [])]
            ollama_ready = True
        except (error.URLError, TimeoutError, ValueError, OSError):
            ollama_ready = False

        def model_present(name: str) -> bool:
            base = name.split(":", 1)[0]
            return any(item == name or item.split(":", 1)[0] == base for item in models)

        return {
            "status": "ready" if (index_ready and ollama_ready) else "degraded",
            "index_ready": index_ready,
            "ollama_ready": ollama_ready,
            "chat_model": competition.MODEL,
            "chat_model_ready": model_present(competition.MODEL),
            "embedding_model": competition.EMBED_MODEL,
            "embedding_model_ready": model_present(competition.EMBED_MODEL),
            "local_processing": True,
        }

    def sources(self) -> list[str]:
        services, _ = self._ensure_ready()
        return sorted(services)

    # ---------------------------------------------------------------- session
    def reset(self, session_id: str) -> None:
        """Forget one conversation only; other sessions are untouched."""
        self._sessions.pop(session_id, None)

    def _session(self, session_id: str) -> dict:
        return self._sessions.setdefault(
            session_id,
            {"history": [], "previous_service": None, "previous_topics": None},
        )

    # ----------------------------------------------------------------- answer
    def answer(self, question: str, session_id: str) -> dict:
        """Answer one message, mirroring ``main.main()`` per-message behaviour."""
        question = (question or "").strip()
        if not question:
            raise ValueError("empty question")

        _, retriever = self._ensure_ready()
        started = time.perf_counter()

        # Single concurrency: the local model must never be called twice at once.
        with self._lock:
            session = self._session(session_id)
            history: list[tuple[str, str]] = session["history"]

            if competition.should_block_sensitive(question):
                answer = SENSITIVE_REPLY
                contexts: list[str] = []
                history.append((SENSITIVE_HISTORY_PLACEHOLDER, answer))
            else:
                ids = competition.route(question, session["previous_service"])
                topics = competition.choose_topics(question, session["previous_topics"])
                retrieval_question = question
                if session["previous_topics"] and not competition.detected_topics(question):
                    retrieval_question += "\nالموضوع السابق: " + "، ".join(session["previous_topics"])
                pool = competition.retrieve_contexts(retrieval_question, ids, retriever)
                missing_details = competition.missing_fee_details(question, ids, history)
                try:
                    answer, contexts = competition.generate(question, pool, history, missing_details)
                except (error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                    raise EngineUnavailable(ENGINE_UNAVAILABLE_AR) from exc
                except (ValueError, KeyError):
                    answer, contexts = MODEL_UNAVAILABLE_REPLY, []
                history.append((question, answer))
                if len(ids) == 1:
                    session["previous_service"] = ids[0]
                    session["previous_topics"] = topics

            if len(history) > MAX_HISTORY_TURNS:
                del history[:-MAX_HISTORY_TURNS]

        return {
            "answer": answer,
            "contexts": list(contexts),
            "session_id": session_id,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
