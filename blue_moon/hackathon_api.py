"""Launch the FastAPI backend and bundled web UI."""

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hackathon_rag.api import validate_bind_security  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="NovaCortex local RAG web server")
    parser.add_argument("--host", default=os.getenv("HACKATHON_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("HACKATHON_API_PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    validate_bind_security(args.host, os.getenv("HACKATHON_API_TOKEN", ""))
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install hackathon_requirements.txt before starting the API") from exc
    uvicorn.run(
        "hackathon_rag.api:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        access_log=True,
        factory=True,
    )


if __name__ == "__main__":
    main()
