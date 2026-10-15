"""Run with: python -m unittest test_competition_main.py (from project root)."""

import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


class FakeCollection:
    """Disk-backed stand-in for Chroma, so index wiring can be tested offline."""

    def __init__(self, directory):
        self.path = Path(directory) / "collection.json"

    def _items(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def upsert(self, ids, documents, metadatas, embeddings):
        items = self._items()
        for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            items[chunk_id] = {"document": document, "metadata": metadata, "embedding": embedding}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    def query(self, query_embeddings, n_results, include, where=None):
        items = self._items()
        allowed = set(where["source"]["$in"]) if where else None
        ids = [chunk_id for chunk_id, item in items.items() if allowed is None or item["metadata"]["source"] in allowed]
        return {"ids": [ids[:n_results]]}


class FakePersistentClient:
    def __init__(self, path):
        self.path = path

    def get_or_create_collection(self, name, metadata):
        return FakeCollection(self.path)


FAKE_CHROMA = SimpleNamespace(PersistentClient=FakePersistentClient)


class CompetitionEntryTests(unittest.TestCase):
    def setUp(self):
        self.services = main.parse_services()

    def test_all_services_have_nine_fields(self):
        self.assertEqual(len(self.services), 12)
        for service in self.services.values():
            self.assertEqual(set(service["fields"]), set(main.FIELDS))
        parents, chunks = main.structured_chunks(self.services)
        self.assertEqual((len(parents), len(chunks)), (105, 105))
        self.assertEqual(max(len(chunk.body) for chunk in chunks), 740)
        self.assertTrue(all(chunk.source.endswith(".txt") and chunk.section in main.FIELDS for chunk in chunks))

    def test_vector_index_is_persisted_and_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict("sys.modules", {"chromadb": FAKE_CHROMA}), patch.object(main, "INDEX_DIR", Path(temp) / "index"), patch.object(main, "embed_texts", side_effect=lambda texts, batch_size=16: [[1.0, 0.0] for _ in texts]) as embed:
                first = main.ensure_index(self.services)
                second = main.ensure_index(self.services)
            self.assertEqual(len(first.chunks), 105)
            self.assertEqual(len(second.chunks), 105)
            self.assertEqual(embed.call_count, 1)
            self.assertTrue((Path(temp) / "index" / "chroma" / "collection.json").exists())
            self.assertEqual(json.loads((Path(temp) / "index" / "manifest.json").read_text(encoding="utf-8"))["embedding_model"], main.EMBED_MODEL)

    def test_service_routing_and_followup(self):
        self.assertEqual(main.route("وين أطلع جواز سفر بدل فاقد؟"), ["passport_lost"])
        self.assertEqual(main.route("جواز بدل فاقد للمرة الأولى، كم بدفع؟"), ["passport_lost"])
        self.assertEqual(main.route("جوازي باقي عليه 3 شهور، كم الرسوم؟"), ["passport_renewal"])
        self.assertEqual(
            set(main.route("قديش صورة للبطاقة الذكية لأول مرة، وقديش لتجديد جواز السفر؟")),
            {"id_card_first_issue", "passport_renewal"},
        )
        self.assertEqual(main.route("وبقدر من السفارة؟", "family_book_renewal"), ["family_book_renewal"])
        self.assertEqual(main.route("كم سعر تذكرة الطيران؟"), [])

    def test_comparison_questions_route_to_every_mentioned_service(self):
        cases = {
            "قديش صورة شخصية بحتاج للبطاقة الذكية لأول مرة، وقديش لتجديد جواز السفر؟": {"id_card_first_issue", "passport_renewal"},
            "قديش رسوم إصدار البطاقة الذكية لأول مرة وكم رسوم تجديد جواز السفر؟": {"id_card_first_issue", "passport_renewal"},
            "شو الأوراق المطلوبة لدفتر العائلة بدل فاقد ولجواز السفر بدل فاقد؟": {"family_book_lost", "passport_lost"},
            "وين بقدر أطلع جواز سفر بدل فاقد وبطاقة بدل فاقد؟": {"passport_lost", "id_card_lost"},
            "شو شروط إصدار جواز سفر لأول مرة وتجديد جواز السفر؟": {"passport_first_issue", "passport_renewal"},
            "شو أوراق تجديد الجواز ورسوم تجديد البطاقة؟": {"passport_renewal", "id_card_renewal"},
        }
        for question, expected in cases.items():
            self.assertEqual(set(main.route(question)), expected, question)

    def test_comparison_prompt_asks_for_each_service_and_flattens_answer(self):
        contexts = [
            "إصدار بطاقة الأحوال المدنية الذكية لأول مرة - الوثائق المطلوبة للحصول على الخدمة: x",
            "تجديد جواز السفر العادي - الوثائق المطلوبة للحصول على الخدمة: y",
        ]
        self.assertEqual(main._context_service_titles(contexts), ["إصدار بطاقة الأحوال المدنية الذكية لأول مرة", "تجديد جواز السفر العادي"])
        prompt = main._build_prompt("سؤال", contexts, [])
        self.assertIn("أجب عن كل خدمة على حدة", prompt)
        single_prompt = main._build_prompt("سؤال", contexts[:1], [])
        self.assertNotIn("أجب عن كل خدمة على حدة", single_prompt)
        self.assertEqual(main._flatten_answer({"أ": "1", "ب": "2"}), "أ: 1؛ ب: 2")
        self.assertEqual(main._flatten_answer(["1", {"ب": "2"}]), "1 ب: 2")
        self.assertEqual(main._flatten_answer("نص"), "نص")

    def test_relevant_contexts_keep_service_and_field(self):
        contexts = main.collect_contexts("كم يستغرق تجديد دفتر العائلة؟", self.services, ["family_book_renewal"])
        self.assertEqual(len(contexts), 2)
        self.assertTrue(any("45 دقيقة" in context for context in contexts))
        self.assertTrue(all("تجديد دفتر العائلة" in context for context in contexts))
        lost_id = main.collect_contexts("كم يستغرق إصدار بطاقة بدل فاقد؟", self.services, ["id_card_lost"])
        self.assertTrue(any("20 دقيقة" in context for context in lost_id))
        self.assertTrue(any("أسبوعين" in context for context in lost_id))
        self.assertFalse(any("تجديد جواز السفر" in context for context in lost_id))

    def test_embedded_retrieval_stays_inside_selected_service(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict("sys.modules", {"chromadb": FAKE_CHROMA}), patch.object(main, "INDEX_DIR", Path(temp) / "index"), patch.object(main, "embed_texts", side_effect=lambda texts, batch_size=16: [[1.0, 0.0] for _ in texts]) as embed:
                retriever = main.ensure_index(self.services)
                contexts = main.retrieve_contexts("كم رسوم دفتر العائلة بدل فاقد؟", ["family_book_lost"], retriever)
            self.assertEqual(embed.call_count, 2)  # document batch and query
        self.assertTrue(any("5 دنانير" in item for item in contexts))
        self.assertTrue(any("تعهد" in item for item in contexts))
        self.assertTrue(all("دفتر عائلة بدل فاقد" in item for item in contexts))

    def test_personal_fee_needs_details_but_general_fee_does_not(self):
        self.assertEqual(main.missing_fee_details("ما هي رسوم تجديد جواز السفر؟", ["passport_renewal"], []), [])
        missing = main.missing_fee_details("عمري 17 وجوازي باقي عليه 3 شهور، كم بدفع؟", ["passport_renewal"], [])
        self.assertTrue(any("داخل الأردن" in item for item in missing))
        self.assertFalse(any("كم عمرك" in item for item in missing))
        self.assertFalse(any("بقي على انتهائه" in item for item in missing))
        self.assertEqual(main.missing_fee_details("كم بدفع لتجديد دفتر العائلة المنتهي؟", ["family_book_renewal"], []), [])
        self.assertTrue(main.missing_fee_details("كم بدفع لتجديد دفتر العائلة؟", ["family_book_renewal"], []))
        self.assertTrue(main.missing_fee_details("بدي أجدد دفتر العائلة، كم علي؟", ["family_book_renewal"], []))
        self.assertEqual(main.missing_fee_details("جواز بدل فاقد للمرة الأولى كم بدفع؟", ["passport_lost"], []), [])
        self.assertEqual(main.missing_fee_details("ضاعت بطاقتي للمرة الثانية قبل سنتين، كم بدفع؟", ["id_card_lost"], []), [])

    def test_output_matches_competition_shape(self):
        practice = Path(__file__).with_name("questions.json")
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "answers.json"
            with patch.object(main, "OUTPUT", out), patch.object(main, "ensure_index", return_value=object()), patch.object(main, "retrieve_contexts", side_effect=lambda q, ids, r: main.collect_contexts(q, self.services, ids)), patch.object(main, "generate", side_effect=lambda q, c, h, m: ("إجابة من النموذج", c[:1])), patch.object(main.sys, "argv", ["main.py", str(practice)]):
                self.assertEqual(main.main(), 0)
            source = json.loads(practice.read_text(encoding="utf-8"))
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(result["chats"]), len(source["chats"]))
            for original_chat, answered_chat in zip(source["chats"], result["chats"]):
                self.assertEqual(original_chat["chat_id"], answered_chat["chat_id"])
                for original, answered in zip(original_chat["messages"], answered_chat["messages"]):
                    for key in ("message_id", "question"):
                        self.assertEqual(original[key], answered[key])
                    self.assertIsInstance(answered["answer"], str)
                    self.assertIsInstance(answered["contexts"], list)

    def test_missing_model_citations_do_not_claim_every_context(self):
        response = {"message": {"content": json.dumps({"answer": "جواب بلا إسناد", "used_ids": []}, ensure_ascii=False)}}
        with patch.object(main.request, "urlopen", return_value=io.BytesIO(json.dumps(response, ensure_ascii=False).encode("utf-8"))):
            answer, contexts = main.generate("سؤال", ["خدمة - رسوم: 5 دنانير"], [], [])
        self.assertEqual(answer, "جواب بلا إسناد")
        self.assertEqual(contexts, [])

    def test_offline_http_flow_builds_index_and_answers_all_practice_messages(self):
        def fake_ollama(call, timeout):
            payload = json.loads(call.data.decode("utf-8"))
            if call.full_url.endswith("/api/embed"):
                value = {"embeddings": [[1.0, 0.0] for _ in payload["input"]]}
            else:
                value = {"message": {"content": json.dumps({"answer": "إجابة اختبارية", "used_ids": [1]}, ensure_ascii=False)}}
            return io.BytesIO(json.dumps(value, ensure_ascii=False).encode("utf-8"))

        practice = Path(__file__).with_name("questions.json")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict("sys.modules", {"chromadb": FAKE_CHROMA}), patch.object(main, "INDEX_DIR", root / "index"), patch.object(main, "OUTPUT", root / "answers.json"), patch.object(main.request, "urlopen", side_effect=fake_ollama), patch.object(main.sys, "argv", ["main.py", str(practice)]):
                self.assertEqual(main.main(), 0)
            answers = json.loads((root / "answers.json").read_text(encoding="utf-8"))
            self.assertTrue((root / "index" / "manifest.json").exists())
            self.assertEqual(sum(len(chat["messages"]) for chat in answers["chats"]), 6)
            self.assertTrue(answers["chats"][2]["messages"][1]["contexts"])
            self.assertEqual(answers["chats"][4]["messages"][0]["contexts"], [])


if __name__ == "__main__":
    unittest.main()
