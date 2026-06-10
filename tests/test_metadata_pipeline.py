from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legal_rag.cli import main
from legal_rag.formatting.submission import format_prediction
from legal_rag.generation.ollama import OllamaConfig
from legal_rag.planner import LegalQueryPlan, PlannedQuery, plan_legal_query
from legal_rag.question_metadata import infer_runtime_metadata, split_questions_for_gold
from legal_rag.schemas.models import PredictedQuestionMetadata, Question, QuestionRunTrace
from legal_rag.verifier import VerificationResult


class MetadataPipelineTest(unittest.TestCase):
    def test_rental_duration_does_not_require_land_component(self) -> None:
        metadata = infer_runtime_metadata(
            "Công ty nhỏ và vừa được hỗ trợ giá thuê mặt bằng sản xuất trong thời gian tối đa là bao lâu?"
        )

        self.assertEqual(metadata.requested_components, ["thời hạn"])

    def test_required_components_have_question_source_spans(self) -> None:
        metadata = infer_runtime_metadata("Cơ sở ươm tạo được hỗ trợ gì về thuế và đất đai?")

        required = [item for item in metadata.component_requirements if item.requirement == "required"]

        self.assertTrue(required)
        self.assertTrue(all(item.source_span for item in required))
        self.assertEqual({item.name for item in required}, set(metadata.requested_components))

    def test_list_question_does_not_imply_support_policy_role(self) -> None:
        metadata = infer_runtime_metadata("Phạm vi đăng ký thuế bao gồm những nội dung cụ thể nào?")

        self.assertEqual(metadata.question_type, "list")
        self.assertNotIn("support_policy", metadata.target_norm_roles)

    def test_overpaid_penalty_money_does_not_request_penalty_amount(self) -> None:
        metadata = infer_runtime_metadata("Khi công ty nộp thừa tiền thuế và tiền phạt thì được xử lý bằng những cách nào?")

        self.assertNotIn("mức phạt", metadata.requested_components)
        self.assertNotIn("penalty", metadata.target_norm_roles)

    def test_penalty_amount_question_requires_penalty_component(self) -> None:
        metadata = infer_runtime_metadata("Công ty không khai báo máy móc thì bị phạt bao nhiêu?")

        self.assertIn("mức phạt", metadata.requested_components)
        requirement = next(item for item in metadata.component_requirements if item.name == "mức phạt")
        self.assertEqual(requirement.requirement, "required")
        self.assertTrue(requirement.source_span)

    def test_responsibility_is_not_authority(self) -> None:
        metadata = infer_runtime_metadata("Bộ Tài chính có trách nhiệm hướng dẫn những nội dung gì về thuế?")

        self.assertIn("trách nhiệm", metadata.requested_components)
        self.assertNotIn("authority", metadata.target_norm_roles)

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
        self.assertIn("Luật Thủ đô", metadata.domain_anchors)
        self.assertIn("Luật Thủ đô", metadata.governing_doc_hints)

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

    def test_planner_repairs_invalid_json_once(self) -> None:
        repaired = {
            "intent": "authority",
            "question_scope": "unknown",
            "normalized_question": "Cơ quan có thẩm quyền xử phạt thuế",
            "question_type": "authority",
            "answer_shape": "document_pointer",
            "legal_terms": ["thẩm quyền", "xử phạt", "thuế"],
            "legal_facets": ["thẩm quyền", "thuế"],
            "requested_components": ["thẩm quyền"],
            "target_norm_roles": ["authority"],
            "entities": {"subjects": [], "actions": [], "objects": [], "conditions": [], "amounts_or_deadlines": []},
            "target_doc_ids": [],
            "target_doc_aliases": [],
            "target_article_labels": [],
            "queries": [{"kind": "original", "text": "Cơ quan nào xử phạt thuế?", "purpose": "preserve user wording"}],
            "filters": {"doc_types": [], "must_include_terms": [], "should_include_terms": []},
            "retrieval_bias": "authority_articles",
            "needs_guidance_docs": True,
            "multi_hop_targets": ["Nghị định"],
            "missing_facts": [],
            "confidence": 0.8,
        }

        with patch("legal_rag.planner.request_ollama_chat", side_effect=["{bad json", json.dumps(repaired, ensure_ascii=False)]) as mocked:
            plan = plan_legal_query("Cơ quan nào xử phạt thuế?", OllamaConfig())

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(plan.intent, "authority")
        self.assertEqual(plan.retrieval_bias, "authority_articles")

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
