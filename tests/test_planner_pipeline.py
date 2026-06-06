from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from legal_rag.corpus.ingest import ingest_corpus, write_articles_jsonl
from legal_rag.generation.evidence import build_evidence_blocks, evidence_answer_from_json, articles_from_used_evidence
from legal_rag.planner import LegalQueryPlan, PlannedQuery, legal_query_plan_from_json
from legal_rag.retrieval import BM25Index
from legal_rag.retrieval.hybrid import HybridRetrievalConfig, HybridRetriever, build_hybrid_index
from legal_rag.schemas.models import ArticleNode
from legal_rag.verifier import verify_used_evidence_answer


class FakeEmbedder:
    def encode(self, texts, return_dense=True, return_sparse=True, return_colbert_vecs=False):
        return {
            "dense_vecs": [[float(i + 1)] * 1024 for i, _text in enumerate(texts)],
            "lexical_weights": [{str(i + 1): 1.0} for i, _text in enumerate(texts)],
        }


class FakeReranker:
    def compute_score(self, pairs, normalize=True):
        return [1.0 - i * 0.01 for i, _pair in enumerate(pairs)]


class CountingQdrantClient:
    def __init__(self, points=None):
        self.points = points or []
        self.queries = []
        self.upserted = []

    def collection_exists(self, collection):
        return False

    def delete_collection(self, collection):
        raise AssertionError("unexpected delete")

    def create_collection(self, collection, vectors_config, sparse_vectors_config):
        return None

    def upsert(self, collection_name, points):
        self.upserted.extend(points)

    def query_points(self, collection_name, query, using, with_payload, limit):
        self.queries.append(using)
        return type("Result", (), {"points": self.points[:limit]})()


class PlannerPipelineTest(unittest.TestCase):
    def test_retrieval_import_is_not_order_dependent(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "from legal_rag.retrieval import BM25Index, retrieve_articles"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_planner_rejects_hallucinated_doc_id(self) -> None:
        data = self._plan_json("Doanh nghiệp được hỗ trợ khi nào?")
        data["target_doc_ids"] = ["99/2099/QH99"]

        with self.assertRaisesRegex(Exception, "planner_doc_id_not_in_question"):
            legal_query_plan_from_json(data, "Doanh nghiệp được hỗ trợ khi nào?")

    def test_planner_accepts_article_label_only_when_present_in_question(self) -> None:
        data = self._plan_json("Theo Điều 4, doanh nghiệp được hỗ trợ khi nào?")
        data["target_article_labels"] = ["Điều 4"]

        plan = legal_query_plan_from_json(data, "Theo Điều 4, doanh nghiệp được hỗ trợ khi nào?")

        self.assertEqual(plan.target_article_labels, ["Điều 4"])

    def test_tax_land_plan_sets_guidance_flag(self) -> None:
        data = self._plan_json("Cơ sở ươm tạo được hỗ trợ gì về thuế và đất đai?")
        data["intent"] = "tax_land"
        data["needs_guidance_docs"] = True

        plan = legal_query_plan_from_json(data, "Cơ sở ươm tạo được hỗ trợ gì về thuế và đất đai?")

        self.assertTrue(plan.needs_guidance_docs)
        self.assertEqual(plan.intent, "tax_land")

    def test_hybrid_search_with_plan_queries_dense_and_sparse_for_each_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            articles, index_path, _articles_path = self._build_inputs(root)
            client = CountingQdrantClient([{"payload": article.to_dict(), "score": 0.5} for article in articles])
            retriever = HybridRetriever(
                articles,
                BM25Index.load(index_path),
                HybridRetrievalConfig(rerank_top_k=5, final_top_k=2),
                qdrant_client=client,
                embedder=FakeEmbedder(),
                reranker=FakeReranker(),
            )
            plan = LegalQueryPlan(
                intent="condition",
                question_scope="single_article",
                normalized_question="điều kiện hỗ trợ doanh nghiệp",
                queries=[
                    PlannedQuery("original", "Doanh nghiệp được hỗ trợ khi nào?"),
                    PlannedQuery("legal_terms", "điều kiện hỗ trợ doanh nghiệp"),
                ],
            )

            hits = retriever.search_with_plan("Doanh nghiệp được hỗ trợ khi nào?", plan, top_k=2)

            self.assertTrue(hits)
            self.assertEqual(client.queries.count("dense"), 2)
            self.assertEqual(client.queries.count("sparse"), 2)

    def test_micro_chunk_build_promotes_to_parent_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_text = "Điều 4. Điều kiện hỗ trợ\n" + "Nội dung hỗ trợ doanh nghiệp. " * 220
            article = ArticleNode(
                article_key="01/2020/QH14|Luật 01/2020/QH14 Luật X|Điều 4",
                doc_id="01/2020/QH14",
                doc_type="Luật",
                title_for_submission="Luật 01/2020/QH14 Luật X",
                article_label="Điều 4",
                article_title="Điều kiện hỗ trợ",
                text=long_text,
                metadata={"chunk_type": "article"},
            )
            articles_path = root / "articles.jsonl"
            write_articles_jsonl([article], articles_path)
            client = CountingQdrantClient()

            report = build_hybrid_index(
                articles_path,
                HybridRetrievalConfig(collection="test", embedding_cache_dir=str(root / "cache")),
                qdrant_client=client,
                embedder=FakeEmbedder(),
            )

            self.assertGreater(report.points, report.articles)
            payloads = [point["payload"] if isinstance(point, dict) else point.payload for point in client.upserted]
            self.assertTrue(any(payload["metadata"].get("parent_article_key") == article.article_key for payload in payloads))

    def test_answer_formatter_uses_only_selected_evidence(self) -> None:
        article4 = self._article("Điều 4")
        article5 = self._article("Điều 5")
        blocks = build_evidence_blocks([article4, article5])
        answer = evidence_answer_from_json(
            {
                "answer": "Doanh nghiệp được hỗ trợ theo Điều 4.",
                "used_evidence_ids": ["E1"],
                "insufficient_evidence": False,
                "support_map": [],
            },
            blocks,
        )

        used = articles_from_used_evidence(blocks, answer.used_evidence_ids)

        self.assertEqual([article.relevant_article for article in used], [article4.relevant_article])

    def test_verifier_detects_ambiguous_same_article_label(self) -> None:
        a = self._article("Điều 5", doc_id="01/2020/QH14")
        b = self._article("Điều 5", doc_id="02/2020/QH14")

        result = verify_used_evidence_answer("Căn cứ Điều 5, doanh nghiệp phải nộp hồ sơ.", [a, b], [a, b])

        self.assertFalse(result.ok)
        self.assertIn("ambiguous_article_label:điều 5", result.issues)

    @staticmethod
    def _plan_json(question: str) -> dict:
        return {
            "intent": "condition",
            "question_scope": "single_article",
            "normalized_question": question,
            "legal_terms": ["doanh nghiệp", "hỗ trợ"],
            "entities": {"subjects": [], "actions": [], "conditions": [], "amounts_or_deadlines": []},
            "target_doc_ids": [],
            "target_doc_aliases": [],
            "target_article_labels": [],
            "queries": [{"kind": "original", "text": question, "purpose": "preserve user wording"}],
            "filters": {"doc_types": [], "must_include_terms": [], "should_include_terms": []},
            "needs_guidance_docs": False,
            "multi_hop_targets": [],
            "missing_facts": [],
            "confidence": 0.7,
        }

    @staticmethod
    def _article(label: str, doc_id: str = "01/2020/QH14") -> ArticleNode:
        return ArticleNode(
            article_key=f"{doc_id}|Luật {doc_id} Luật X|{label}",
            doc_id=doc_id,
            doc_type="Luật",
            title_for_submission=f"Luật {doc_id} Luật X",
            article_label=label,
            article_title="Hồ sơ",
            text=f"{label}. Hồ sơ\nDoanh nghiệp nộp hồ sơ.",
        )

    @staticmethod
    def _build_inputs(root: Path):
        raw_doc = {
            "doc_id": "01/2020/QH14",
            "doc_type": "Luật",
            "trich_yeu": "Luật X",
            "title_for_submission": "Luật 01/2020/QH14 Luật X",
            "raw_text": "Điều 4. Điều kiện hỗ trợ\nDoanh nghiệp được hỗ trợ khi đáp ứng tiêu chí.\n\nĐiều 5. Hồ sơ\nHồ sơ gồm đơn đề nghị.",
        }
        corpus_path = root / "corpus.json"
        corpus_path.write_text(json.dumps([raw_doc], ensure_ascii=False), encoding="utf-8")
        articles, warnings = ingest_corpus(corpus_path)
        if warnings:
            raise AssertionError(warnings)
        articles_path = root / "articles.jsonl"
        write_articles_jsonl(articles, articles_path)
        index_path = root / "index.json"
        BM25Index(articles).save(index_path)
        return articles, index_path, articles_path


if __name__ == "__main__":
    unittest.main()
