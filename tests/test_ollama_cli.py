from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from legal_rag.cli import _batch_run_state, main
from legal_rag.generation.ollama import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaConfig,
    OllamaError,
    request_ollama_chat,
)
from legal_rag.corpus.ingest import ingest_corpus, write_articles_jsonl
from legal_rag.formatting.submission import validate_package_manifest
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
    def test_native_ollama_request_uses_context_keep_alive_and_thinking(self) -> None:
        response = FakeResponse({"message": {"content": '{"ok":true}'}})
        config = OllamaConfig(
            url="http://127.0.0.1:11434/api/chat",
            response_format_json=True,
            keep_alive="30m",
            think=True,
            num_ctx=16384,
            retries=0,
        )

        with patch("legal_rag.generation.ollama.urlopen", return_value=response) as mocked:
            content = request_ollama_chat("system", "user", config, json_schema={"type": "object"})

        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(content, '{"ok":true}')
        self.assertTrue(payload["think"])
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertEqual(payload["format"], {"type": "object"})

    def test_native_empty_thinking_response_recovers_without_thinking(self) -> None:
        empty = FakeResponse(
            {
                "message": {"content": "", "thinking": "Đang suy luận nhưng chưa kịp trả lời"},
                "done_reason": "length",
                "eval_count": 32,
            }
        )
        recovered = FakeResponse({"message": {"content": '{"ok":true}'}, "done_reason": "stop"})
        config = OllamaConfig(
            url="http://127.0.0.1:11434/api/chat",
            think=True,
            max_tokens=32,
            retries=0,
            empty_content_retries=1,
        )

        with patch("legal_rag.generation.ollama.urlopen", side_effect=[empty, recovered]) as mocked:
            content = request_ollama_chat("system", "user", config, json_schema={"type": "object"})

        first = json.loads(mocked.call_args_list[0].args[0].data.decode("utf-8"))
        second = json.loads(mocked.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(content, '{"ok":true}')
        self.assertTrue(first["think"])
        self.assertFalse(second["think"])
        self.assertEqual(second["options"]["num_predict"], 512)
        self.assertEqual(first["format"], second["format"])

    def test_native_empty_response_reports_ollama_metadata_after_recovery(self) -> None:
        empty = FakeResponse(
            {
                "message": {"content": "", "thinking": "unfinished"},
                "done_reason": "length",
                "eval_count": 16,
            }
        )
        config = OllamaConfig(
            url="http://127.0.0.1:11434/api/chat",
            think=True,
            retries=0,
            empty_content_retries=1,
        )

        with patch("legal_rag.generation.ollama.urlopen", side_effect=[empty, empty]):
            with self.assertRaisesRegex(
                OllamaError,
                r"ollama_empty_answer:done_reason=length:eval_count=16:thinking_chars=10:recovery=true",
            ):
                request_ollama_chat("system", "user", config)

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
            self.assertEqual(data[0]["answer"], "Doanh nghiệp được hỗ trợ. Căn cứ Điều 4 01/2020/QH14.")
            self.assertNotIn("Theo các căn cứ đã truy hồi", data[0]["answer"])
            manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["generator_backend"], "ollama")
            self.assertEqual(manifest["planner_backend"], "ollama")
            self.assertEqual(manifest["model"], "qwen3:8b-q8_0")

    def test_run_batch_repairs_invalid_answer_json_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output = root / "results.json"
            invalid = FakeResponse({"choices": [{"message": {"content": "not-json"}}]})
            payloads = [
                FakeResponse(planner_payload()),
                invalid,
                FakeResponse(answer_payload("Doanh nghiệp được hỗ trợ khi đáp ứng tiêu chí.")),
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
            self.assertEqual(mocked.call_count, 3)

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

    def test_model_citation_is_rebuilt_from_used_evidence(self) -> None:
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

            self.assertEqual(code, 0)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data[0]["answer"], "Câu trả lời. Căn cứ Điều 4 01/2020/QH14.")
            self.assertNotIn("Điều 99", data[0]["answer"])

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
            run_state = _batch_run_state(
                questions_path=str(questions_path),
                index_path=str(index_path),
                articles_path="data/normalized/articles.jsonl",
                backend="bm25_exact",
                top_k=5,
                model=DEFAULT_OLLAMA_MODEL,
                ollama_url=DEFAULT_OLLAMA_URL,
                config={},
            )
            (output.parent / f".{output.stem}.run.json").write_text(
                json.dumps(run_state, ensure_ascii=False),
                encoding="utf-8",
            )
            output.with_suffix(".failed.jsonl").write_text(
                json.dumps({"id": 2, "question": "Hồ sơ đề nghị gồm những gì?", "stage": "verifier"}) + "\n",
                encoding="utf-8",
            )

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
            self.assertEqual(data[1]["answer"], "Hồ sơ. Căn cứ Điều 5 01/2020/QH14.")
            self.assertFalse(output.with_suffix(".failed.jsonl").exists())

    def test_run_batch_refuses_resume_when_fingerprint_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output = root / "results.json"
            output.with_suffix(".progress.jsonl").write_text("{}\n", encoding="utf-8")
            (output.parent / f".{output.stem}.run.json").write_text(
                json.dumps({"fingerprint": "different"}),
                encoding="utf-8",
            )

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

    def test_run_batch_can_seed_a_new_fingerprinted_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            seed = root / "legacy.progress.jsonl"
            seed.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "question": "Doanh nghiệp được hỗ trợ khi nào?",
                        "answer": "Câu trả lời cũ.",
                        "relevant_docs": [],
                        "relevant_articles": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "results_v2.json"

            code = main(
                [
                    "run_batch",
                    "--questions",
                    str(questions_path),
                    "--index",
                    str(index_path),
                    "--output",
                    str(output),
                    "--resume-from",
                    str(seed),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((root / ".results_v2.run.json").exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["answer"], "Câu trả lời cũ.")

    def test_package_manifest_rejects_quarantined_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "results.json"
            result.write_text("[]", encoding="utf-8")
            result.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "generator_backend": "ollama",
                        "planner_backend": "ollama",
                        "model": "qwen3:8b-q8_0",
                        "planner_model": "qwen3:8b-q8_0",
                        "ollama_url": "http://127.0.0.1:11434/v1/chat/completions",
                        "verifier_issues": [],
                        "failed": 1,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            issues = validate_package_manifest(result)

            self.assertIn("manifest_has_failed_questions:1", issues)


if __name__ == "__main__":
    unittest.main()
