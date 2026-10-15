"""Deterministic LangGraph workflow for the local document assistant."""

import inspect
import time
from typing import Callable, Dict, List, TypedDict

from .cleaning import clean_text
from .models import Answer, ParentSection, RetrievedChunk


class RAGState(TypedDict, total=False):
    question: str
    response_language: str
    retrieved: List[RetrievedChunk]
    parents: Dict[str, ParentSection]
    answer: Answer
    timings_ms: Dict[str, int]
    workflow: List[str]
    source_scope: List[str]


class DeterministicRAGWorkflow:
    """A fixed, inspectable workflow; it does not let an LLM choose its path."""

    def __init__(
        self,
        retrieve: Callable[[str], List[RetrievedChunk]],
        load_parents: Callable[[], Dict[str, ParentSection]],
        answer: Callable[[str, List[RetrievedChunk], Dict[str, ParentSection], str], Answer],
    ):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph is required for the hackathon workflow. "
                "Install hackathon_requirements.txt first."
            ) from exc

        self._retrieve_callback = retrieve
        self._retrieve_accepts_scope = len(inspect.signature(retrieve).parameters) >= 2
        self._load_parents_callback = load_parents
        self._answer_callback = answer

        graph = StateGraph(RAGState)
        graph.add_node("validate_question", self._validate_question)
        graph.add_node("retrieve_evidence", self._retrieve_evidence)
        graph.add_node("load_parent_context", self._load_parent_context)
        graph.add_node("generate_grounded_answer", self._generate_grounded_answer)
        graph.add_edge(START, "validate_question")
        graph.add_edge("validate_question", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "load_parent_context")
        graph.add_edge("load_parent_context", "generate_grounded_answer")
        graph.add_edge("generate_grounded_answer", END)
        self._graph = graph.compile()

    @staticmethod
    def _record(state: RAGState, name: str, elapsed_ms: int) -> dict:
        timings = dict(state.get("timings_ms", {}))
        timings[name] = elapsed_ms
        workflow = list(state.get("workflow", []))
        workflow.append(name)
        return {"timings_ms": timings, "workflow": workflow}

    def _validate_question(self, state: RAGState) -> dict:
        started = time.perf_counter()
        question = clean_text(state.get("question", ""))
        if not question:
            raise ValueError("Question cannot be empty")
        update = self._record(state, "validate_question", int((time.perf_counter() - started) * 1000))
        update["question"] = question
        return update

    def _retrieve_evidence(self, state: RAGState) -> dict:
        started = time.perf_counter()
        retrieved = (
            self._retrieve_callback(state["question"], state.get("source_scope", []))
            if self._retrieve_accepts_scope else
            self._retrieve_callback(state["question"])
        )
        update = self._record(state, "retrieve_evidence", int((time.perf_counter() - started) * 1000))
        update["retrieved"] = retrieved
        return update

    def _load_parent_context(self, state: RAGState) -> dict:
        started = time.perf_counter()
        parents = self._load_parents_callback()
        update = self._record(state, "load_parent_context", int((time.perf_counter() - started) * 1000))
        update["parents"] = parents
        return update

    def _generate_grounded_answer(self, state: RAGState) -> dict:
        started = time.perf_counter()
        answer = self._answer_callback(
            state["question"],
            state.get("retrieved", []),
            state.get("parents", {}),
            state.get("response_language", "ar"),
        )
        update = self._record(state, "generate_grounded_answer", int((time.perf_counter() - started) * 1000))
        answer.timings_ms = update["timings_ms"]
        answer.workflow = update["workflow"]
        update["answer"] = answer
        return update

    def invoke(
        self,
        question: str,
        response_language: str = "ar",
        source_scope: List[str] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        state = self._graph.invoke({
            "question": question,
            "response_language": response_language,
            "timings_ms": {},
            "workflow": [],
            "source_scope": list(source_scope or []),
        })
        result = state["answer"]
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result
