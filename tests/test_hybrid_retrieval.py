from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from legal_rag.corpus.ingest import ingest_corpus, write_articles_jsonl
from legal_rag.retrieval import BM25Index
from legal_rag.retrieval.hybrid import (
    HybridRetrievalConfig,
    HybridRetrievalError,
    HybridRetriever,
    build_hybrid_index,
    fuse_ranked_lists,
)
from legal_rag.schemas.models import ArticleNode


RAW_DOC = {
    "doc_id": "01/2020/QH14",
    "doc_type": "Luật",
    "trich_yeu": "Luật X",
    "title_for_submission": "Luật 01/2020/QH14 Luật X",
    "raw_text": (
        "Điều 4. Điều kiện hỗ trợ\n"
        "Doanh nghiệp được hỗ trợ khi đáp ứng tiêu chí.\n\n"
        "Điều 5. Hồ sơ\n"
        "Hồ sơ gồm đơn đề nghị và tài liệu liên quan.\n\n"
        "Điều 6. Kinh phí\n"
        "Kinh phí hỗ trợ được bố trí từ ngân sách."
    ),
}


class FakeEmbedder:
    def encode(self, texts, return_dense=True, return_sparse=True, return_colbert_vecs=False):
        return {
            "dense_vecs": [[float(i + 1)] * 1024 for i, _text in enumerate(texts)],
            "lexical_weights": [{str(i + 1): 1.0} for i, _text in enumerate(texts)],
        }


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        self.text_count = 0

    def encode(self, texts, return_dense=True, return_sparse=True, return_colbert_vecs=False):
        self.text_count += len(texts)
        return super().encode(texts, return_dense=return_dense, return_sparse=return_sparse, return_colbert_vecs=return_colbert_vecs)


class ExplodingEmbedder:
    def encode(self, texts, return_dense=True, return_sparse=True, return_colbert_vecs=False):
        raise AssertionError("complete cache should avoid encode")


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = []

    def compute_score(self, pairs, normalize=True):
        self.pairs = pairs
        return self.scores[: len(pairs)]


class FakeQdrantClient:
    def __init__(self, points=None):
        self.points = points or []
        self.created = None
        self.upserted = []

    def collection_exists(self, collection):
        return False

    def delete_collection(self, collection):
        raise AssertionError("delete should not be called for missing fake collection")

    def create_collection(self, collection, vectors_config, sparse_vectors_config):
        self.created = {
            "collection": collection,
            "vectors_config": vectors_config,
            "sparse_vectors_config": sparse_vectors_config,
        }

    def upsert(self, collection_name, points):
        self.upserted.extend(points)

    def query_points(self, collection_name, query, using, with_payload, limit):
        selected = self.points[:limit] if using == "dense" else list(reversed(self.points[:limit]))
        return type("Result", (), {"points": selected})()


def _read_cache_lines(path: Path) -> list[str]:
    if ".gz" in path.suffixes:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read().splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def _write_cache_text(path: Path, text: str) -> None:
    if ".gz" in path.suffixes:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
        return
    path.write_text(text, encoding="utf-8")


class HybridRetrievalTest(unittest.TestCase):
    def test_fusion_keeps_exact_article_candidate_at_top(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            articles, _index_path, _articles_path = self._build_inputs(Path(tmp))
            exact = articles[1]
            exact.score = 320.0
            vector_hit = articles[2]
            vector_hit.score = 0.8

            fused = fuse_ranked_lists([[exact], [vector_hit, exact]], limit=2)

            self.assertEqual(fused[0].article_key, exact.article_key)

    def test_hybrid_reranker_only_scores_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            articles, index_path, articles_path = self._build_inputs(root)
            points = [{"payload": article.to_dict(), "score": 0.5} for article in articles]
            reranker = FakeReranker([0.1, 0.9])
            retriever = HybridRetriever(
                articles,
                BM25Index.load(index_path),
                HybridRetrievalConfig(rerank_top_k=2, final_top_k=1),
                qdrant_client=FakeQdrantClient(points),
                embedder=FakeEmbedder(),
                reranker=reranker,
            )

            hits = retriever.search("Hồ sơ đề nghị theo Điều 5 gồm những gì?", top_k=1)

            self.assertEqual(len(reranker.pairs), 2)
            self.assertEqual(len(hits), 1)
            self.assertIn(hits[0].article_label, {"Điều 4", "Điều 5", "Điều 6"})

    def test_build_hybrid_index_writes_qdrant_payloads_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            articles, _index_path, articles_path = self._build_inputs(root)
            client = FakeQdrantClient()
            config = HybridRetrievalConfig(embedding_cache_dir=str(root / "cache"), collection="test_law")

            report = build_hybrid_index(articles_path, config, qdrant_client=client, embedder=FakeEmbedder())

            self.assertEqual(report.articles, len(articles))
            self.assertEqual(client.created["collection"], "test_law")
            self.assertGreaterEqual(len(client.upserted), len(articles))
            first_point = client.upserted[0]
            first_payload = first_point["payload"] if isinstance(first_point, dict) else first_point.payload
            self.assertEqual(first_payload["article_label"], "Điều 4")
            self.assertEqual(first_payload["metadata"]["chunk_type"], "article")
            self.assertTrue(Path(report.cache_path).exists())

    def test_hybrid_payload_compacts_span_metadata_and_caps_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            article = ArticleNode(
                article_key="01/2020/QH14|Luật 01/2020/QH14 Luật X|Điều 4",
                doc_id="01/2020/QH14",
                doc_type="Luật",
                title_for_submission="Luật 01/2020/QH14 Luật X",
                article_label="Điều 4",
                article_title="Điều kiện hỗ trợ",
                text="Điều 4. Điều kiện hỗ trợ\n" + "Nội dung rất dài. " * 100,
                metadata={
                    "chunk_type": "article",
                    "clause_nodes": [
                        {
                            "span_key": "Điều 4|Khoản 1",
                            "level": "clause",
                            "label": "Khoản 1",
                            "text": "1. Doanh nghiệp được hỗ trợ.",
                            "point_nodes": [],
                        }
                    ],
                },
            )
            articles_path = root / "articles.jsonl"
            write_articles_jsonl([article], articles_path)
            client = FakeQdrantClient()

            build_hybrid_index(
                articles_path,
                HybridRetrievalConfig(
                    embedding_cache_dir=str(root / "cache"),
                    collection="test_law",
                    max_payload_text_chars=80,
                    upsert_batch_size=1,
                ),
                qdrant_client=client,
                embedder=FakeEmbedder(),
            )

            payloads = [point["payload"] if isinstance(point, dict) else point.payload for point in client.upserted]
            self.assertTrue(any(payload["metadata"].get("node_type") == "clause" for payload in payloads))
            for payload in payloads:
                self.assertLessEqual(len(payload["text"]), 80)
                self.assertNotIn("clause_nodes", payload["metadata"])
                self.assertNotIn("parent_text", payload["metadata"])

    def test_hybrid_index_streams_cache_and_resumes_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _articles, _index_path, articles_path = self._build_inputs(root)
            config = HybridRetrievalConfig(embedding_cache_dir=str(root / "cache"), collection="test_law", batch_size=1)

            first_embedder = CountingEmbedder()
            first_report = build_hybrid_index(
                articles_path,
                config,
                qdrant_client=FakeQdrantClient(),
                embedder=first_embedder,
            )
            self.assertEqual(first_embedder.text_count, first_report.points)

            cache_path = Path(first_report.cache_path)
            rows = _read_cache_lines(cache_path)
            _write_cache_text(cache_path, rows[0] + "\n")

            resume_client = FakeQdrantClient()
            resume_embedder = CountingEmbedder()
            build_hybrid_index(
                articles_path,
                config,
                qdrant_client=resume_client,
                embedder=resume_embedder,
            )

            self.assertEqual(resume_embedder.text_count, first_report.points - 1)
            self.assertEqual(len(resume_client.upserted), first_report.points)

            complete_client = FakeQdrantClient()
            build_hybrid_index(
                articles_path,
                config,
                qdrant_client=complete_client,
                embedder=ExplodingEmbedder(),
            )
            self.assertEqual(len(complete_client.upserted), first_report.points)

    def test_build_hybrid_index_refuses_large_embedded_qdrant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _articles, _index_path, articles_path = self._build_inputs(root)

            with self.assertRaisesRegex(HybridRetrievalError, "embedded_qdrant_too_large"):
                build_hybrid_index(
                    articles_path,
                    HybridRetrievalConfig(
                        embedding_cache_dir=str(root / "cache"),
                        collection="test_law",
                        qdrant_path=str(root / "embedded_qdrant"),
                        max_embedded_points=1,
                    ),
                    qdrant_client=FakeQdrantClient(),
                    embedder=FakeEmbedder(),
                )

    def test_graph_expansion_uses_guidance_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_docs = [
                {
                    "doc_id": "01/2020/QH14",
                    "doc_type": "Luật",
                    "trich_yeu": "Luật X",
                    "title_for_submission": "Luật 01/2020/QH14 Luật X",
                    "raw_text": "Điều 4. Điều kiện hỗ trợ\nDoanh nghiệp được hỗ trợ khi đáp ứng tiêu chí.",
                },
                {
                    "doc_id": "02/2020/NĐ-CP",
                    "doc_type": "Nghị định",
                    "trich_yeu": "Quy định chi tiết Luật X",
                    "title_for_submission": "Nghị định 02/2020/NĐ-CP Quy định chi tiết Luật X",
                    "raw_text": "Nghị định số 02/2020/NĐ-CP quy định chi tiết Điều 4 Luật số 01/2020/QH14.\n\nĐiều 5. Hồ sơ\nHồ sơ gồm đơn đề nghị.",
                },
            ]
            corpus_path = root / "corpus.json"
            corpus_path.write_text(json.dumps(raw_docs, ensure_ascii=False), encoding="utf-8")
            articles, warnings = ingest_corpus(corpus_path)
            if warnings:
                raise AssertionError(warnings)
            articles_path = root / "articles.jsonl"
            write_articles_jsonl(articles, articles_path)
            index_path = root / "index.json"
            BM25Index(articles).save(index_path)
            client = FakeQdrantClient([{"payload": article.to_dict(), "score": 0.5} for article in articles])
            retriever = HybridRetriever(
                articles,
                BM25Index.load(index_path),
                HybridRetrievalConfig(rerank_top_k=6, final_top_k=3),
                qdrant_client=client,
                embedder=FakeEmbedder(),
                reranker=FakeReranker([1.0] * 6),
            )

            from legal_rag.planner import LegalQueryPlan, PlannedQuery

            plan = LegalQueryPlan(
                intent="procedure",
                question_scope="single_article",
                normalized_question="hồ sơ hỗ trợ theo Luật X",
                question_type="procedure",
                answer_shape="procedure_steps",
                legal_terms=["hồ sơ", "hỗ trợ"],
                requested_components=["hồ sơ"],
                target_norm_roles=["procedure"],
                queries=[PlannedQuery("original", "Hồ sơ hỗ trợ theo Luật X gồm những gì?")],
                needs_guidance_docs=True,
            )

            hits = retriever.search_with_plan("Hồ sơ hỗ trợ theo Luật X gồm những gì?", plan, top_k=3)

            self.assertTrue(any(hit.doc_id == "02/2020/NĐ-CP" for hit in hits))
            self.assertIn("02/2020/NĐ-CP", retriever.reverse_guides_by_doc_id.get("01/2020/QH14", set()))

    @staticmethod
    def _build_inputs(root: Path):
        corpus_path = root / "corpus.json"
        corpus_path.write_text(json.dumps([RAW_DOC], ensure_ascii=False), encoding="utf-8")
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
