from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legal_rag.corpus.ingest import ingest_corpus, write_articles_jsonl
from legal_rag.formatting.submission import (
    format_prediction,
    package_submission,
    validate_submission,
    write_predictions,
)
from legal_rag.generation import generate_grounded_answer
from legal_rag.retrieval import BM25Index, retrieve_articles
from legal_rag.schemas.models import ArticleNode, Question
from legal_rag.utils.text import extract_article_labels
from legal_rag.verifier import verify_prediction_evidence


RAW_DOC = {
    "doc_id": "01/2020/QH14",
    "doc_type": "Luật",
    "trich_yeu": "Luật X",
    "title_for_submission": "Luật 01/2020/QH14 Luật X",
    "raw_text": "Điều 4. Điều kiện hỗ trợ\nDoanh nghiệp nhỏ và vừa được hỗ trợ khi đáp ứng tiêu chí.\n\nĐiều 5. Hồ sơ\nHồ sơ gồm đơn đề nghị và tài liệu liên quan.",
}


class CorePipelineTest(unittest.TestCase):
    def test_extract_article_patterns(self) -> None:
        labels = extract_article_labels("Căn cứ khoản 2 Điều 4 và Điều 5 của Luật X.")
        self.assertEqual(labels, ["Điều 4", "Điều 5"])

    def test_ingest_parse_retrieve_format_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_path = root / "corpus.json"
            corpus_path.write_text(json.dumps([RAW_DOC], ensure_ascii=False), encoding="utf-8")

            articles, warnings = ingest_corpus(corpus_path)
            self.assertEqual(warnings, [])
            self.assertEqual(len(articles), 2)
            self.assertEqual(articles[0].article_label, "Điều 4")

            index = BM25Index(articles)
            hits = retrieve_articles(index, "Doanh nghiệp nhỏ và vừa được hỗ trợ theo điều kiện nào?", top_k=2)
            self.assertTrue(hits)

            answer = generate_grounded_answer("?", hits)
            verification = verify_prediction_evidence(answer, hits)
            self.assertTrue(verification.ok, verification.issues)

            prediction = format_prediction(Question(id=1, question="?"), answer, hits)
            output = root / "results.json"
            write_predictions([prediction], output)
            questions = root / "questions.json"
            questions.write_text(json.dumps([{"id": 1, "question": "?"}], ensure_ascii=False), encoding="utf-8")
            self.assertEqual(validate_submission(output, questions), [])
            zip_path = root / "submission.zip"
            package_submission(output, zip_path)
            self.assertTrue(zip_path.exists())

            normalized = root / "articles.jsonl"
            write_articles_jsonl(articles, normalized)
            self.assertTrue(normalized.read_text(encoding="utf-8").strip())

    def test_generated_answer_masks_unretrieved_internal_article_refs(self) -> None:
        article = ArticleNode(
            article_key="01/2020/QH14|Điều 1",
            doc_id="01/2020/QH14",
            doc_type="Luật",
            title_for_submission="Luật 01/2020/QH14 Luật X",
            article_label="Điều 1",
            article_title="Phạm vi điều chỉnh",
            text="Điều 1. Phạm vi điều chỉnh\nHồ sơ áp dụng theo Điều 6 của Luật này.",
        )

        answer = generate_grounded_answer("?", [article])
        verification = verify_prediction_evidence(answer, [article])

        self.assertTrue(verification.ok, verification.issues)
        self.assertNotIn("Điều 6", answer)


if __name__ == "__main__":
    unittest.main()
