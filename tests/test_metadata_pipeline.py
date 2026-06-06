from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legal_rag.cli import main
from legal_rag.formatting.submission import format_prediction
from legal_rag.planner import LegalQueryPlan, PlannedQuery
from legal_rag.question_metadata import infer_runtime_metadata, split_questions_for_gold
from legal_rag.schemas.models import PredictedQuestionMetadata, Question, QuestionRunTrace
from legal_rag.verifier import VerificationResult


class MetadataPipelineTest(unittest.TestCase):
    def test_runtime_metadata_infers_list_shape(self) -> None:
        metadata = infer_runtime_metadata(
            "Luật Thủ đô quy định những chính sách đặc thù nào?",
            intent="support_policy",
            legal_terms=["Luật Thủ đô", "chính sách đặc thù"],
        )

        self.assertEqual(metadata.question_type, "list")
        self.assertEqual(metadata.answer_shape, "list_items")
        self.assertEqual(metadata.retrieval_bias, "content_articles")
        self.assertTrue(metadata.legal_facets)

    def test_split_questions_for_gold_creates_holdout(self) -> None:
        questions = [
            Question(id=1, question="Doanh nghiệp được hỗ trợ khi nào?"),
            Question(id=2, question="Thủ tục đăng ký ra sao?"),
            Question(id=3, question="Mức xử phạt thế nào?"),
            Question(id=4, question="Những chính sách nào được áp dụng?"),
        ]
        predicted = {question.id: infer_runtime_metadata(question.question) for question in questions}

        tune, holdout = split_questions_for_gold(questions, predicted_by_id=predicted)

        self.assertTrue(tune)
        self.assertTrue(holdout)
        self.assertEqual(sorted(question.id for question in tune + holdout), [1, 2, 3, 4])

    def test_prepare_gold_metadata_writes_scaffolds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions_path = root / "questions.json"
            questions_path.write_text(
                json.dumps(
                    [
                        {"id": 1, "question": "Doanh nghiệp được hỗ trợ khi nào?"},
                        {"id": 2, "question": "Thủ tục đăng ký ra sao?"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output_dir = root / "gold"
            plan = LegalQueryPlan(
                intent="condition",
                question_scope="single_article",
                normalized_question="Doanh nghiệp được hỗ trợ khi nào?",
                question_type="condition",
                answer_shape="conditions_list",
                legal_terms=["doanh nghiệp", "hỗ trợ"],
                legal_facets=["hỗ trợ"],
                queries=[PlannedQuery("original", "Doanh nghiệp được hỗ trợ khi nào?")],
                retrieval_bias="content_articles",
                confidence=0.8,
            )

            with patch("legal_rag.cli.plan_legal_query", return_value=plan):
                code = main(
                    [
                        "prepare_gold_metadata",
                        "--questions",
                        str(questions_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "dev_gold.jsonl").exists())
            self.assertTrue((output_dir / "holdout_gold.jsonl").exists())
            rows = (output_dir / "dev_gold.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(rows)
            first = json.loads(rows[0])
            self.assertIn("question_id", first)
            self.assertIn("question_type", first)
            self.assertIn("gold_relevant_articles", first)

    def test_eval_pipeline_reports_metadata_and_task_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions_path = root / "questions.json"
            expected_path = root / "expected.jsonl"
            questions_path.write_text(
                json.dumps([{"id": 1, "question": "Doanh nghiệp được hỗ trợ khi nào?"}], ensure_ascii=False),
                encoding="utf-8",
            )
            expected_path.write_text(
                json.dumps(
                    {
                        "question_id": 1,
                        "intent": "condition",
                        "question_type": "condition",
                        "answer_shape": "conditions_list",
                        "needs_guidance_docs": False,
                        "gold_relevant_docs": ["01/2020/QH14|Luật 01/2020/QH14 Luật X"],
                        "gold_relevant_articles": ["01/2020/QH14|Luật 01/2020/QH14 Luật X|Điều 4"],
                        "notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            index_path = root / "index.json"
            index_path.write_text(json.dumps({"articles": []}, ensure_ascii=False), encoding="utf-8")
            articles_path = root / "articles.jsonl"
            articles_path.write_text("", encoding="utf-8")
            plan = LegalQueryPlan(
                intent="condition",
                question_scope="single_article",
                normalized_question="Doanh nghiệp được hỗ trợ khi nào?",
                question_type="condition",
                answer_shape="conditions_list",
                legal_terms=["doanh nghiệp", "hỗ trợ"],
                legal_facets=["hỗ trợ"],
                queries=[PlannedQuery("original", "Doanh nghiệp được hỗ trợ khi nào?")],
                retrieval_bias="content_articles",
                confidence=0.8,
            )
            predicted = format_prediction(
                Question(id=1, question="Doanh nghiệp được hỗ trợ khi nào?"),
                "Doanh nghiệp được hỗ trợ theo Điều 4.",
                [],
            )
            predicted.relevant_docs = ["01/2020/QH14|Luật 01/2020/QH14 Luật X"]
            predicted.relevant_articles = ["01/2020/QH14|Luật 01/2020/QH14 Luật X|Điều 4"]
            trace = QuestionRunTrace(
                id=1,
                question="Doanh nghiệp được hỗ trợ khi nào?",
                predicted_metadata=PredictedQuestionMetadata(
                    intent="condition",
                    question_type="condition",
                    answer_shape="conditions_list",
                    needs_guidance_docs=False,
                    legal_facets=["hỗ trợ"],
                    retrieval_bias="content_articles",
                    planned_queries=["Doanh nghiệp được hỗ trợ khi nào?"],
                    confidence=0.8,
                ),
                candidate_counts={"bm25": 1},
                reranked_evidence=["01/2020/QH14|Luật 01/2020/QH14 Luật X|Điều 4"],
                used_evidence_ids=["E1"],
                verifier_issues=[],
                final_relevant_docs=predicted.relevant_docs,
                final_relevant_articles=predicted.relevant_articles,
            )

            with patch("legal_rag.cli._load_retrieval_backend", return_value=(object(), {"retrieval_backend": "hybrid_qdrant"})), patch(
                "legal_rag.cli.plan_legal_query", return_value=plan
            ), patch(
                "legal_rag.cli._search_articles",
                return_value=[],
            ), patch(
                "legal_rag.cli._answer_question",
                return_value=(predicted, trace, VerificationResult(ok=True, issues=[])),
            ):
                with patch("builtins.print") as mocked_print:
                    code = main(
                        [
                            "eval_pipeline",
                            "--questions",
                            str(questions_path),
                            "--expected",
                            str(expected_path),
                            "--index",
                            str(index_path),
                            "--articles",
                            str(articles_path),
                        ]
                    )

            self.assertEqual(code, 0)
            payload = json.loads(mocked_print.call_args_list[-1].args[0])
            self.assertIn("metadata_metrics", payload)
            self.assertIn("task_metrics", payload)


if __name__ == "__main__":
    unittest.main()
