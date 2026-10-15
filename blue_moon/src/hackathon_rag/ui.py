"""Simple Streamlit demonstration UI for the local document assistant."""

from dataclasses import replace
from pathlib import Path

from .config import HackathonConfig
from .pipeline import HackathonRAG


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install streamlit from hackathon_requirements.txt") from exc

    st.set_page_config(page_title="Local Document Assistant", page_icon="📚", layout="wide")
    st.title("📚 Local Document Assistant")
    st.caption("Grounded answers from local documents through Ollama")

    defaults = HackathonConfig()
    with st.sidebar:
        st.header("Configuration")
        documents = Path(st.text_input("Documents directory", str(defaults.documents_dir)))
        index = Path(st.text_input("Index directory", str(defaults.index_dir)))
        chat_model = st.text_input("Chat model", defaults.chat_model)
        embedding_model = st.text_input("Embedding model", defaults.embedding_model)
        config = replace(
            defaults,
            documents_dir=documents,
            index_dir=index,
            chat_model=chat_model,
            embedding_model=embedding_model,
        )
        rag = HackathonRAG(config)

        source_scope = []
        try:
            indexed_sources = rag.available_sources()
            source_scope = st.multiselect(
                "Search only in selected files (optional)",
                indexed_sources,
                help="Leave empty to search all indexed documents.",
            )
        except RuntimeError:
            pass

        if st.button("Check environment", use_container_width=True):
            st.json(rag.doctor())
        if st.button("Preview chunks", use_container_width=True):
            with st.spinner("Extracting and checking chunks..."):
                try:
                    report = rag.preview(index.parent / "chunk_preview.json", 20)
                    st.success(f"{report['chunks']} chunks from {report['inventory']['supported_files']} files")
                    st.json(report["chunk_lengths"])
                    extraction = report.get("extraction", {})
                    excluded = extraction.get("quarantined", 0) + extraction.get("restricted", 0)
                    if excluded:
                        st.error(f"{excluded} unsafe or restricted file(s) excluded from the index")
                    with st.expander("Extraction and security report"):
                        st.json(extraction)
                except Exception as exc:
                    st.error(str(exc))
        if st.button("Build index", type="primary", use_container_width=True):
            with st.spinner("Embedding documents and building the index..."):
                try:
                    st.json(rag.ingest(reset=True))
                    st.success("Index is ready")
                except Exception as exc:
                    st.error(str(exc))

    question = st.chat_input("Ask a question about the documents")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                try:
                    result = rag.ask(question, source_scope=source_scope)
                    if result.privacy_notice:
                        st.info(result.privacy_notice)
                    st.write(result.answer)
                    if result.answer_mode == "extractive":
                        st.info("The model answer was not safely verifiable, so exact relevant passages are shown instead.")
                    for warning in result.evidence_warnings:
                        st.warning(warning)
                    if result.workflow == ["privacy_guard"]:
                        st.info("Privacy guard stopped the request before retrieval or model processing")
                    elif result.supported:
                        st.success("Supported by the provided documents")
                    else:
                        st.warning("The documents did not provide enough support")
                    if result.citations:
                        st.subheader("Sources")
                        for citation in result.citations:
                            location = citation["source"]
                            if citation.get("page") is not None:
                                location += f" — page {citation['page']}"
                            if citation.get("section"):
                                location += f" — {citation['section']}"
                            with st.expander(location):
                                if citation.get("reason"):
                                    st.caption(citation["reason"])
                                if citation.get("quote"):
                                    st.code(citation["quote"], language=None)
                    with st.expander("Retrieval details"):
                        st.json(result.retrieved)
                    with st.expander("Workflow timings"):
                        st.write(" → ".join(result.workflow))
                        st.json(result.timings_ms)
                    st.caption(f"Total answer latency: {result.latency_ms} ms")
                    if result.request_id:
                        st.caption(f"Request ID: {result.request_id}")
                except Exception as exc:
                    st.error(str(exc))
