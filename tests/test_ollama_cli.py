from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from legal_rag.cli import main
from legal_rag.corpus.ingest import ingest_corpus, write_articles_jsonl
from legal_rag.retrieval import BM25Index

RAW_DOC = {
    "doc_id": "01/2020/QH14",
    "doc_type": "Luật",
    "trich_yeu": "Luật X",
    "title_for_submission": "Luật 01/2020/QH14 Luật X",
    "raw_text": "Điều 4. Điều kiện hỗ trợ\nDoanh nghiệp được hỗ trợ khi đáp ứng tiêu chí.",
}


def planner_payload(question: str = "Doanh nghiệp được hỗ trợ khi nào?") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "intent": "condition",
                            "question_scope": "single_article",
                            "normalized_question": question,
                            "legal_terms": ["doanh nghiệp", "hỗ trợ", "điều kiện"],
                            "entities": {
                                "subjects": ["doanh nghiệp"],
                                "actions": ["hỗ trợ"],
                                "conditions": [],
                                "amounts_or_deadlines": [],
                            },
                            "target_doc_ids": [],
                            "target_doc_aliases": [],
                            "target_article_labels": [],
                            "queries": [
                                {"kind": "original", "text": question, "purpose": "preserve user wording"},
                                {"kind": "legal_terms", "text": "doanh nghiệp hỗ trợ điều kiện", "purpose": "BM25/exact legal terms"},
                            ],
                            "filters": {"doc_types": [], "must_include_terms": [], "should_include_terms": []},
                            "needs_guidance_docs": False,
                            "multi_hop_targets": [],
                            "missing_facts": [],
                            "confidence": 0.8,
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }


def answer_payload(answer: str, evidence_ids: list[str] | None = None) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": answer,
                            "used_evidence_ids": evidence_ids or ["E1"],
                            "insufficient_evidence": False,
                            "support_map": [],
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class OllamaCliTest(unittest.TestCase):
    def test_run_batch_uses_ollama_answer_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output = root / "results.json"
            payloads = [FakeResponse(planner_payload()), FakeResponse(answer_payload("Doanh nghiệp được hỗ trợ theo Điều 4."))]

            with patch("legal_rag.generation.ollama.urlopen", side_effect=payloads) as mocked:
                code = main(
                    [
                        "run_batch",
                        "--questions",
                        str(questions_path),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output),
                        "--model",
                        "qwen3:8b-q8_0",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(mocked.called)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data[0]["answer"], "Doanh nghiệp được hỗ trợ theo Điều 4.")
            self.assertNotIn("Theo các căn cứ đã truy hồi", data[0]["answer"])
            manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["generator_backend"], "ollama")
            self.assertEqual(manifest["planner_backend"], "ollama")
            self.assertEqual(manifest["model"], "qwen3:8b-q8_0")

    def test_run_batch_refuses_to_write_when_ollama_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output = root / "results.json"

            with patch("legal_rag.generation.ollama.urlopen", side_effect=URLError("offline")):
                code = main(
                    [
                        "run_batch",
                        "--questions",
                        str(questions_path),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".manifest.json").exists())

    def test_run_batch_refuses_to_write_without_retrieved_articles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "empty_index.json"
            BM25Index([]).save(index_path)
            questions_path = root / "questions.json"
            questions_path.write_text(
                json.dumps([{"id": 1, "question": "Không có căn cứ thì sao?"}], ensure_ascii=False),
                encoding="utf-8",
            )
            output = root / "results.json"

            with patch("legal_rag.generation.ollama.urlopen", return_value=FakeResponse(planner_payload("Không có căn cứ thì sao?"))):
                code = main(
                    [
                        "run_batch",
                        "--questions",
                        str(questions_path),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".manifest.json").exists())

    def test_verifier_issue_is_recorded_and_blocks_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output = root / "results.json"
            payloads = [FakeResponse(planner_payload()), FakeResponse(answer_payload("Câu trả lời căn cứ Điều 99."))]

            with patch("legal_rag.generation.ollama.urlopen", side_effect=payloads):
                code = main(
                    [
                        "run_batch",
                        "--questions",
                        str(questions_path),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".manifest.json").exists())

    def test_debug_retrieval_output_cannot_be_packaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output = root / "results.json"

            code = main(
                [
                    "debug_retrieval",
                    "--questions",
                    str(questions_path),
                    "--index",
                    str(index_path),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(code, 0)
            manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["generator_backend"], "template_debug")
            package_code = main(["package_submission", "--input", str(output), "--output", str(root / "submission.zip")])
            self.assertEqual(package_code, 1)

    @staticmethod
    def _build_inputs(root: Path) -> tuple[Path, Path]:
        corpus_path = root / "corpus.json"
        corpus_path.write_text(json.dumps([RAW_DOC], ensure_ascii=False), encoding="utf-8")
        articles, warnings = ingest_corpus(corpus_path)
        if warnings:
            raise AssertionError(warnings)
        articles_path = root / "articles.jsonl"
        write_articles_jsonl(articles, articles_path)
        index_path = root / "index.json"
        BM25Index(articles).save(index_path)
        questions_path = root / "questions.json"
        questions_path.write_text(
            json.dumps([{"id": 1, "question": "Doanh nghiệp được hỗ trợ khi nào?"}], ensure_ascii=False),
            encoding="utf-8",
        )
        return index_path, questions_path

    def test_submit_creates_timestamped_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output_dir = root / "output"
            payloads = [FakeResponse(planner_payload()), FakeResponse(answer_payload("Doanh nghiệp được hỗ trợ theo Điều 4."))]

            with patch("legal_rag.generation.ollama.urlopen", side_effect=payloads):
                code = main(
                    [
                        "submit",
                        "--questions",
                        str(questions_path),
                        "--index",
                        str(index_path),
                        "--output-dir",
                        str(output_dir),
                        "--model",
                        "qwen3:8b-q8_0",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(output_dir.exists())
            results_files = [p for p in output_dir.glob("results_2*.json") if "manifest" not in p.name]
            self.assertEqual(len(results_files), 1)
            canonical = output_dir / "results.json"
            self.assertTrue(canonical.exists())
            zip_files = list(output_dir.glob("submission_*.zip"))
            self.assertEqual(len(zip_files), 1)

            data = json.loads(canonical.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["id"], 1)

    def test_run_batch_creates_progress_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output = root / "results.json"
            payloads = [FakeResponse(planner_payload()), FakeResponse(answer_payload("Doanh nghiệp được hỗ trợ theo Điều 4."))]

            with patch("legal_rag.generation.ollama.urlopen", side_effect=payloads):
                code = main(
                    [
                        "run_batch",
                        "--questions",
                        str(questions_path),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 0)
            progress = output.with_suffix(".progress.jsonl")
            self.assertTrue(progress.exists())
            lines = [line for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["id"], 1)

    def test_run_batch_resumes_from_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Build corpus with 2 articles
            raw_doc_2 = {
                "doc_id": "01/2020/QH14",
                "doc_type": "Luật",
                "trich_yeu": "Luật X",
                "title_for_submission": "Luật 01/2020/QH14 Luật X",
                "raw_text": "Điều 4. Điều kiện hỗ trợ\nDoanh nghiệp được hỗ trợ khi đáp ứng tiêu chí.\n\nĐiều 5. Hồ sơ\nHồ sơ gồm đơn đề nghị và tài liệu liên quan.",
            }
            corpus_path = root / "corpus.json"
            corpus_path.write_text(json.dumps([raw_doc_2], ensure_ascii=False), encoding="utf-8")
            articles, _ = ingest_corpus(corpus_path)
            write_articles_jsonl(articles, root / "articles.jsonl")
            index_path = root / "index.json"
            BM25Index(articles).save(index_path)
            questions_path = root / "questions.json"
            questions_path.write_text(
                json.dumps(
                    [
                        {"id": 1, "question": "Doanh nghiệp được hỗ trợ khi nào?"},
                        {"id": 2, "question": "Hồ sơ đề nghị gồm những gì?"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output = root / "results.json"

            # Pre-seed progress file with question 1 already done
            progress = output.with_suffix(".progress.jsonl")
            existing = {
                "id": 1,
                "question": "Doanh nghiệp được hỗ trợ khi nào?",
                "answer": "Đã trả lời trước đó, theo Điều 4.",
                "relevant_docs": ["01/2020/QH14|Luật 01/2020/QH14 Luật X"],
                "relevant_articles": ["01/2020/QH14|Luật 01/2020/QH14 Luật X|Điều 4"],
            }
            progress.write_text(json.dumps(existing, ensure_ascii=False) + "\n", encoding="utf-8")

            payloads = [
                FakeResponse(planner_payload("Hồ sơ đề nghị gồm những gì?")),
                FakeResponse(answer_payload("Hồ sơ theo Điều 5.")),
            ]
            with patch("legal_rag.generation.ollama.urlopen", side_effect=payloads) as mocked:
                code = main(
                    [
                        "run_batch",
                        "--questions",
                        str(questions_path),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 0)
            # Ollama should only be called for question 2 planner + answer, not question 1.
            self.assertEqual(mocked.call_count, 2)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 2)
            # Question 1 should have the pre-seeded answer
            self.assertEqual(data[0]["answer"], "Đã trả lời trước đó, theo Điều 4.")
            # Question 2 should have the new Ollama answer
            self.assertEqual(data[1]["answer"], "Hồ sơ theo Điều 5.")


if __name__ == "__main__":
    unittest.main()
