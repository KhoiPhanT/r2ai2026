from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from legal_rag.cli import main
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
            payload = {"choices": [{"message": {"content": "Doanh nghiệp được hỗ trợ theo Điều 4."}}]}

            with patch("legal_rag.generation.ollama.urlopen", return_value=FakeResponse(payload)) as mocked:
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

    def test_verifier_issue_is_recorded_and_package_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, questions_path = self._build_inputs(root)
            output = root / "results.json"
            payload = {"choices": [{"message": {"content": "Câu trả lời căn cứ Điều 99."}}]}

            with patch("legal_rag.generation.ollama.urlopen", return_value=FakeResponse(payload)):
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
            manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["verifier_issues"])
            self.assertIn("manifest_has_verifier_issues", "\n".join(validate_package_manifest(output)))
            package_code = main(["package_submission", "--input", str(output), "--output", str(root / "submission.zip")])
            self.assertEqual(package_code, 1)

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


if __name__ == "__main__":
    unittest.main()
