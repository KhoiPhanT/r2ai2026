from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legal_rag.corpus.ingest import ingest_corpus, write_articles_jsonl
from legal_rag.retrieval import BM25Index
from legal_rag.retrieval.hybrid import (
    HybridRetrievalConfig,
    HybridRetriever,
    build_hybrid_index,
    fuse_ranked_lists,
)


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
            self.assertEqual(len(client.upserted), len(articles))
            first_point = client.upserted[0]
            first_payload = first_point["payload"] if isinstance(first_point, dict) else first_point.payload
            self.assertEqual(first_payload["article_label"], "Điều 4")
            self.assertEqual(first_payload["metadata"]["chunk_type"], "article")
            self.assertTrue(Path(report.cache_path).exists())

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
