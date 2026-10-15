"""FastAPI adapter that serves the Arabic chat UI on top of the SAME engine
used by ``python main.py questions.json``.

All answering is delegated to ``competition_engine.CompetitionEngine``, a thin
adapter over the tested competition code in the repository-root ``main.py``.
No routing, retrieval, prompt, fee or format logic is duplicated here; this
module only exposes HTTP endpoints and optional loopback/token security.
"""

import asyncio
import hmac
import os
import uuid
from pathlib import Path

from .competition_engine import CompetitionEngine, EngineUnavailable


WEB_DIR = Path(__file__).with_name("web")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

_engine: CompetitionEngine | None = None


def validate_bind_security(host: str, token: str) -> None:
    """Refuse an unauthenticated network deployment by default."""
    if host.strip().casefold() not in LOOPBACK_HOSTS and not token.strip():
        raise RuntimeError(
            "HACKATHON_API_TOKEN is required when the API is exposed beyond localhost."
        )


def _engine_instance() -> CompetitionEngine:
    global _engine
    if _engine is None:
        _engine = CompetitionEngine()
    return _engine


def create_app():
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - startup guidance only
        raise RuntimeError(
            "The local UI requires fastapi and uvicorn from hackathon_requirements.txt"
        ) from exc

    api_token = os.getenv("HACKATHON_API_TOKEN", "").strip()
    max_concurrent = max(1, int(os.getenv("HACKATHON_MAX_CONCURRENT_QUESTIONS", "1")))
    generation_slots = asyncio.Semaphore(max_concurrent)

    class ChatRequest(BaseModel):
        question: str = Field(min_length=1, max_length=4000)
        session_id: str | None = Field(default=None, max_length=80)

    class ResetRequest(BaseModel):
        session_id: str | None = Field(default=None, max_length=80)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if not api_token:
            return
        expected = f"Bearer {api_token}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid access token")

    app = FastAPI(
        title="مساعد الخدمات الحكومية المحلي",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    async def health():
        return await run_in_threadpool(_engine_instance().health)

    @app.get("/api/sources", dependencies=[Depends(authorize)])
    async def sources():
        try:
            return {"sources": await run_in_threadpool(_engine_instance().sources)}
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/chat", dependencies=[Depends(authorize)])
    async def chat(payload: ChatRequest):
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="السؤال فارغ.")
        session_id = (payload.session_id or "").strip() or uuid.uuid4().hex
        engine = _engine_instance()
        async with generation_slots:
            try:
                return await run_in_threadpool(engine.answer, question, session_id)
            except EngineUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/chat/reset", dependencies=[Depends(authorize)])
    async def reset(payload: ResetRequest):
        engine = _engine_instance()
        if payload.session_id:
            engine.reset(payload.session_id)
        return {"session_id": uuid.uuid4().hex, "reset": True}

    return app
