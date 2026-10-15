"""Offline tests for the shared competition engine and the chat API.

Run with: python -m unittest test_competition_api.py (from the project root).

These tests never touch Ollama: they use the real, pure-Python routing helpers
from ``main`` and patch only the two network-touching functions
(``retrieve_contexts`` and ``generate``).  That is also what proves the UI and
``main.py`` share one implementation: the engine calls ``main``'s functions.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import main  # noqa: E402
import hackathon_rag.api as api  # noqa: E402
from hackathon_rag.competition_engine import (  # noqa: E402
    CompetitionEngine,
    EngineUnavailable,
)


def fake_engine() -> CompetitionEngine:
    """An engine whose index is already loaded, so no Ollama call is needed."""
    engine = CompetitionEngine()
    engine._retriever = object()
    engine._services = {}
    return engine


class SharedEngineTests(unittest.TestCase):
    def test_answer_delegates_to_main_generate_and_keeps_history(self):
        engine = fake_engine()
        calls = []

        def recorded_generate(question, contexts, history, missing_details=None):
            calls.append((question, list(history)))
            return ("جواب: " + question, ["ctx:" + question])

        with patch.object(main, "retrieve_contexts", return_value=[]), \
                patch.object(main, "generate", side_effect=recorded_generate):
            first = engine.answer("كم رسوم جواز بدل فاقد؟", "s1")
            second = engine.answer("طيب والصور كم؟", "s1")

        self.assertEqual(first["answer"], "جواب: كم رسوم جواز بدل فاقد؟")
        self.assertEqual(second["contexts"], ["ctx:طيب والصور كم؟"])
        self.assertEqual([c[0] for c in calls],
                         ["كم رسوم جواز بدل فاقد؟", "طيب والصور كم؟"])
        # The follow-up is answered in the same session, so it sees turn 1.
        self.assertEqual(calls[1][1][0][0], "كم رسوم جواز بدل فاقد؟")
        self.assertEqual(set(first), {"answer", "contexts", "session_id", "latency_ms"})

    def test_sessions_do_not_leak_into_each_other(self):
        engine = fake_engine()
        with patch.object(main, "retrieve_contexts", return_value=[]), \
                patch.object(main, "generate", return_value=("ok", [])):
            engine.answer("كم رسوم جواز بدل فاقد؟", "a")
            engine.answer("كم رسوم دفتر العائلة؟", "b")
        self.assertEqual(set(engine._sessions), {"a", "b"})
        engine.reset("a")
        self.assertNotIn("a", engine._sessions)
        self.assertIn("b", engine._sessions)

    def test_sensitive_message_is_blocked_without_calling_the_model(self):
        engine = fake_engine()
        with patch.object(main, "generate") as generate:
            result = engine.answer("كلمة المرور هي 123456", "s")
            generate.assert_not_called()
        self.assertEqual(result["contexts"], [])
        self.assertIn("خصوصيتك", result["answer"])

    def test_out_of_scope_question_returns_empty_contexts(self):
        engine = fake_engine()
        with patch.object(main, "retrieve_contexts", return_value=[]), \
                patch.object(main, "generate", return_value=("الوثائق لا تغطيها", [])):
            result = engine.answer("كم سعر تذكرة الطائرة إلى باريس؟", "s")
        self.assertEqual(result["contexts"], [])

    def test_engine_downtime_raises_arabic_error(self):
        engine = fake_engine()
        with patch.object(main, "retrieve_contexts", return_value=[]), \
                patch.object(main, "generate", side_effect=error.URLError("down")):
            with self.assertRaises(EngineUnavailable) as ctx:
                engine.answer("ما هي رسوم الجواز؟", "s")
        self.assertIn("Ollama", str(ctx.exception))


class FakeEngine:
    def __init__(self):
        self.reset_calls = []
        self.health_payload = {
            "status": "ready", "index_ready": True, "ollama_ready": True,
            "chat_model": "command-r7b-arabic:7b", "chat_model_ready": True,
            "embedding_model": "embeddinggemma", "embedding_model_ready": True,
            "local_processing": True,
        }

    def health(self):
        return self.health_payload

    def sources(self):
        return ["passport_first_issue.txt"]

    def answer(self, question, session_id):
        return {
            "answer": "ج:" + question,
            "contexts": ["سياق"],
            "session_id": session_id,
            "latency_ms": 12,
        }

    def reset(self, session_id):
        self.reset_calls.append(session_id)


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.fake = FakeEngine()
        api._engine = self.fake
        self.client = TestClient(api.create_app())

    def tearDown(self):
        api._engine = None

    def test_root_serves_arabic_chat_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("مساعد الخدمات الحكومية", response.text)
        self.assertNotIn("http://", response.text.split("</head>")[0])

    def test_static_assets_are_served_locally(self):
        response = self.client.get("/assets/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("session_id", response.text)

    def test_health_reports_readiness(self):
        body = self.client.get("/api/health").json()
        self.assertTrue(body["index_ready"])
        self.assertTrue(body["ollama_ready"])

    def test_chat_response_shape_is_exact(self):
        response = self.client.post("/api/chat", json={"question": "مرحبا", "session_id": "s1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"answer", "contexts", "session_id", "latency_ms"})
        self.assertEqual(response.json()["session_id"], "s1")

    def test_chat_generates_a_session_when_missing(self):
        body = self.client.post("/api/chat", json={"question": "مرحبا"}).json()
        self.assertTrue(body["session_id"])

    def test_blank_question_is_rejected(self):
        response = self.client.post("/api/chat", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_new_conversation_resets_only_that_session(self):
        response = self.client.post("/api/chat/reset", json={"session_id": "s1"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["reset"])
        self.assertEqual(self.fake.reset_calls, ["s1"])

    def test_engine_downtime_returns_arabic_503(self):
        class Down(FakeEngine):
            def answer(self, question, session_id):
                raise EngineUnavailable("تعذر الاتصال بمحرك Ollama المحلي.")

        api._engine = Down()
        response = self.client.post("/api/chat", json={"question": "مرحبا", "session_id": "s"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Ollama", response.json()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
