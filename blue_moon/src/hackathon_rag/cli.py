"""Command-line interface for the local hackathon assistant."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import HackathonConfig
from .evaluation import evaluate
from .pipeline import HackathonRAG


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _show_answer(result) -> None:
    if result.privacy_notice:
        print(f"[Privacy] {result.privacy_notice}\n")
    print(result.answer)
    if result.answer_mode == "extractive":
        print("\n[Extractive fallback: exact source passages were used.]")
    for warning in result.evidence_warnings:
        print(f"\n[Evidence warning] {warning}")
    if result.citations:
        print("\nSources:")
        for citation in result.citations:
            location = citation["source"]
            if citation.get("page") is not None:
                location += f" — page {citation['page']}"
            if citation.get("section"):
                location += f" — {citation['section']}"
            print(f"- {location}")
            if citation.get("quote"):
                print(f"  \"{citation['quote']}\"")
    if not result.supported:
        print("\n[The answer was not fully supported by the provided documents.]")
    if result.request_id:
        print(f"\nRequest ID: {result.request_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local document-grounded RAG assistant")
    parser.add_argument("--documents", type=Path, help="Override the documents directory")
    parser.add_argument("--index", type=Path, help="Override the index directory")
    parser.add_argument("--chat-model", help="Override the Ollama chat model")
    parser.add_argument("--embedding-model", help="Override the Ollama embedding model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check Ollama, models and configured paths")
    preview = subparsers.add_parser("preview", help="Inspect extracted text and chunks without Ollama")
    preview.add_argument("--output", type=Path, default=Path("hackathon_workspace/chunk_preview.json"))
    preview.add_argument("--samples", type=int, default=30)
    ingest = subparsers.add_parser("ingest", help="Load documents and rebuild the local index")
    ingest.add_argument("--no-reset", action="store_true", help="Do not delete the previous index first")
    ask = subparsers.add_parser("ask", help="Ask one question")
    ask.add_argument("question")
    ask.add_argument("--source", action="append", default=[], help="Limit retrieval to this indexed source (repeatable)")
    inspect = subparsers.add_parser("inspect", help="Show retrieved chunks without asking the LLM")
    inspect.add_argument("question")
    inspect.add_argument("--source", action="append", default=[], help="Limit retrieval to this indexed source (repeatable)")
    chat = subparsers.add_parser("chat", help="Start a simple interactive session")
    evaluation = subparsers.add_parser("evaluate", help="Run a JSON evaluation set")
    evaluation.add_argument("cases", type=Path)
    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    config = HackathonConfig()
    overrides = {}
    if args.documents:
        overrides["documents_dir"] = args.documents
    if args.index:
        overrides["index_dir"] = args.index
    if args.chat_model:
        overrides["chat_model"] = args.chat_model
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
    rag = HackathonRAG(replace(config, **overrides))
    try:
        if args.command == "doctor":
            _print(rag.doctor())
        elif args.command == "preview":
            report = rag.preview(args.output, max(5, args.samples))
            summary = {key: value for key, value in report.items() if key != "samples"}
            summary["preview_file"] = str(args.output)
            _print(summary)
        elif args.command == "ingest":
            _print(rag.ingest(reset=not args.no_reset))
        elif args.command == "ask":
            _show_answer(rag.ask(args.question, source_scope=args.source))
        elif args.command == "inspect":
            _print([
                {
                    "source": item.chunk.source,
                    "page": item.chunk.page,
                    "section": item.chunk.section,
                    "score": round(item.score, 6),
                    "vector_rank": item.vector_rank,
                    "keyword_rank": item.keyword_rank,
                    "preview": item.chunk.body[:500],
                }
                for item in rag.retrieve(args.question, source_scope=args.source)
            ])
        elif args.command == "chat":
            print("Local document assistant. Type /exit to stop.")
            while True:
                question = input("\n> ").strip()
                if question.casefold() in {"/exit", "exit", "quit"}:
                    break
                if question:
                    _show_answer(rag.ask(question))
        elif args.command == "evaluate":
            report = evaluate(rag, args.cases)
            _print(report)
            return 0 if report["passed"] == report["total"] else 1
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
