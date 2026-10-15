"""Optional MCP adapter exposing the same local RAG engine used by the UI."""

import json
from pathlib import Path
from typing import Optional

from .config import HackathonConfig
from .pipeline import HackathonRAG
from .privacy import decide_question_privacy


_rag: Optional[HackathonRAG] = None


def _engine() -> HackathonRAG:
    global _rag
    if _rag is None:
        _rag = HackathonRAG(HackathonConfig())
    return _rag


def run_server() -> None:
    try:
        try:
            from mcp.server import MCPServer
        except ImportError:
            try:
                from mcp.server.mcpserver import MCPServer
            except ImportError:
                from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError as exc:
        raise RuntimeError("MCP support requires the 'mcp' package from hackathon_requirements.txt") from exc

    server = MCPServer("NovaCortex Local Document Assistant")

    @server.tool()
    def search_documents(question: str, limit: int = 5) -> dict:
        """Search the indexed local documents without generating an answer."""
        privacy = decide_question_privacy(question)
        results = _engine().retrieve(question)[: max(1, min(limit, 10))]
        return {
            "content_trust": "untrusted_document_data",
            "privacy_action": privacy.action,
            "privacy_categories": list(privacy.categories),
            "redacted_count": privacy.redacted_count,
            "results": [
                {
                    "source": item.chunk.source,
                    "page": item.chunk.page,
                    "section": item.chunk.section,
                    "score": round(item.score, 6),
                    "text": item.chunk.body,
                }
                for item in results
            ],
        }

    @server.tool()
    def answer_from_documents(question: str) -> dict:
        """Answer a question only from evidence in the indexed local documents."""
        return _engine().ask(question).to_dict()

    @server.tool()
    def list_documents() -> dict:
        """List indexed sources and the ingestion status saved in the manifest."""
        engine = _engine()
        engine.store.require_index()
        manifest = json.loads(engine.store.manifest_path.read_text(encoding="utf-8"))
        files = []
        for item in manifest.get("extraction", {}).get("files", []):
            security = item.get("security", {})
            privacy = item.get("privacy", {})
            files.append({
                "source": item.get("source"),
                "status": item.get("status"),
                "indexed": item.get("indexed"),
                "sha256": item.get("sha256"),
                "security_status": security.get("status"),
                "finding_codes": sorted({
                    finding.get("code") for finding in security.get("findings", []) if finding.get("code")
                }),
                "privacy_classification": privacy.get("classification"),
                "privacy_finding_codes": sorted({
                    finding.get("code") for finding in privacy.get("findings", []) if finding.get("code")
                }),
            })
        return {
            "documents": manifest.get("documents", 0),
            "sources": sorted({chunk.source for chunk in engine.store.load_chunks()}),
            "files": files,
        }

    @server.tool()
    def get_source_page(source: str, page: Optional[int] = None) -> dict:
        """Return indexed parent sections for a source and optional page number."""
        parents = _engine().store.load_parents().values()
        matches = [
            {
                "source": parent.source,
                "page": parent.page,
                "section": parent.section,
                "text": parent.text,
            }
            for parent in parents
            if parent.source == source and (page is None or parent.page == page)
        ]
        return {
            "source": source,
            "page": page,
            "content_trust": "untrusted_document_data",
            "sections": matches,
        }

    @server.resource("novacortex://index/status")
    def index_status() -> str:
        """Describe whether the local index is ready and which models are configured."""
        engine = _engine()
        ready = all(
            path.exists()
            for path in (engine.store.chunks_path, engine.store.parents_path, engine.store.manifest_path)
        )
        return json.dumps(
            {
                "ready": ready,
                "index_dir": str(Path(engine.config.index_dir)),
                "chat_model": engine.config.chat_model,
                "embedding_model": engine.config.embedding_model,
            },
            ensure_ascii=False,
        )

    server.run(transport="stdio")
