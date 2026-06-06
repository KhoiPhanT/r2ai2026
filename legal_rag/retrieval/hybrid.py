from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from legal_rag.corpus.ingest import read_articles_jsonl
from legal_rag.planner import LegalQueryPlan
from legal_rag.retrieval.bm25 import BM25Index
from legal_rag.retrieval.pipeline import exact_match, legal_prior_score, retrieve_articles
from legal_rag.schemas.models import ArticleNode

DEFAULT_COLLECTION = "r2ai_law_articles_v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


class HybridRetrievalError(RuntimeError):
    pass


@dataclass(slots=True)
class HybridRetrievalConfig:
    qdrant_url: str = DEFAULT_QDRANT_URL
    qdrant_path: str = ""
    collection: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    embedding_local_path: str = ""
    reranker_local_path: str = ""
    local_files_only: bool = False
    embedding_cache_dir: str = "data/indices/embedding_cache"
    bm25_top_k: int = 80
    vector_top_k: int = 80
    fusion_top_k: int = 50
    rerank_top_k: int = 30
    final_top_k: int = 5
    min_rerank_score: float | None = None
    batch_size: int = 16


@dataclass(slots=True)
class HybridBuildReport:
    collection: str
    articles: int
    cache_path: str
    embedding_model: str
    qdrant_url: str
    qdrant_path: str = ""
    points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "articles": self.articles,
            "cache_path": self.cache_path,
            "embedding_model": self.embedding_model,
            "qdrant_url": self.qdrant_url,
            "qdrant_path": self.qdrant_path,
            "points": self.points,
        }


class HybridRetriever:
    def __init__(
        self,
        articles: list[ArticleNode],
        bm25_index: BM25Index,
        config: HybridRetrievalConfig,
        *,
        qdrant_client: Any | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.articles = articles
        self.bm25_index = bm25_index
        self.config = config
        self._qdrant_client = qdrant_client
        self._embedder = embedder
        self._reranker = reranker
        self.last_trace: dict[str, Any] = {}

    @classmethod
    def load(cls, articles_path: str | Path, bm25_index_path: str | Path, config: HybridRetrievalConfig) -> "HybridRetriever":
        articles = read_articles_jsonl(articles_path)
        return cls(articles, BM25Index.load(bm25_index_path), config)

    def search(self, question: str, top_k: int | None = None) -> list[ArticleNode]:
        final_top_k = top_k or self.config.final_top_k
        candidate_lists: list[list[ArticleNode]] = []

        exact_hits = exact_match(self.articles, question)
        if exact_hits:
            candidate_lists.append(exact_hits)

        bm25_hits = retrieve_articles(
            self.bm25_index,
            question,
            top_k=self.config.bm25_top_k,
            bm25_top_k=self.config.bm25_top_k,
            min_score_ratio=0.0,
        )
        if bm25_hits:
            candidate_lists.append(bm25_hits)

        dense_hits, sparse_hits = self._search_qdrant(question)
        if dense_hits:
            candidate_lists.append(dense_hits)
        if sparse_hits:
            candidate_lists.append(sparse_hits)

        fused = fuse_ranked_lists(candidate_lists, limit=max(self.config.rerank_top_k, final_top_k))
        if not fused:
            self.last_trace = {"candidate_counts": {}, "fused": 0, "reranked": 0, "final": 0}
            return []

        reranked = self._rerank(question, fused[: self.config.rerank_top_k])
        if self.config.min_rerank_score is not None:
            reranked = [hit for hit in reranked if hit.score >= self.config.min_rerank_score]
        final_hits = reranked[:final_top_k]
        self.last_trace = {
            "candidate_counts": {
                "exact": len(exact_hits),
                "bm25": len(bm25_hits),
                "dense": len(dense_hits),
                "sparse": len(sparse_hits),
            },
            "fused": len(fused),
            "reranked": len(reranked),
            "final": len(final_hits),
        }
        return final_hits

    def search_with_plan(self, question: str, plan: LegalQueryPlan, top_k: int | None = None) -> list[ArticleNode]:
        final_top_k = top_k or self.config.final_top_k
        candidate_lists: list[list[ArticleNode]] = []
        candidate_counts: dict[str, int] = {}
        exact_text = " ".join(
            [question, plan.normalized_question, *plan.target_doc_ids, *plan.target_doc_aliases, *plan.target_article_labels]
        )
        exact_hits = _with_trace(exact_match(self.articles, exact_text), "exact", "planner_targets")
        if exact_hits:
            candidate_lists.append(exact_hits)
        candidate_counts["exact"] = len(exact_hits)

        for planned_query in plan.queries:
            query = planned_query.text
            bm25_hits = retrieve_articles(
                self.bm25_index,
                query,
                top_k=self.config.bm25_top_k,
                bm25_top_k=self.config.bm25_top_k,
                min_score_ratio=0.0,
            )
            if bm25_hits:
                candidate_lists.append(_with_trace(bm25_hits, "bm25", planned_query.kind))
            candidate_counts[f"bm25:{planned_query.kind}"] = len(bm25_hits)

            dense_hits, sparse_hits = self._search_qdrant(query)
            if dense_hits:
                candidate_lists.append(_with_trace(dense_hits, "dense", planned_query.kind))
            if sparse_hits:
                candidate_lists.append(_with_trace(sparse_hits, "sparse", planned_query.kind))
            candidate_counts[f"dense:{planned_query.kind}"] = len(dense_hits)
            candidate_counts[f"sparse:{planned_query.kind}"] = len(sparse_hits)

        guidance_hits: list[ArticleNode] = []
        if plan.needs_guidance_docs:
            guidance_hits = self._guidance_hits(question, plan)
            if guidance_hits:
                candidate_lists.append(_with_trace(guidance_hits, "bm25_guidance", "guidance"))
            candidate_counts["guidance"] = len(guidance_hits)

        fused = fuse_ranked_lists(candidate_lists, limit=max(self.config.fusion_top_k, self.config.rerank_top_k, final_top_k))
        if not fused:
            self.last_trace = {"candidate_counts": candidate_counts, "fused": 0, "reranked": 0, "final": 0}
            return []
        rerank_query = " ".join([question, plan.normalized_question, " ".join(plan.legal_terms)])
        reranked = self._rerank(rerank_query, fused[: self.config.rerank_top_k])
        reranked = _apply_retrieval_bias(reranked, plan)
        reranked = _promote_and_dedupe(reranked)
        if self.config.min_rerank_score is not None:
            reranked = [hit for hit in reranked if hit.score >= self.config.min_rerank_score]
        final_hits = reranked[:final_top_k]
        self.last_trace = {
            "candidate_counts": candidate_counts,
            "fused": len(fused),
            "reranked": len(reranked),
            "final": len(final_hits),
        }
        return final_hits

    def _search_qdrant(self, question: str) -> tuple[list[ArticleNode], list[ArticleNode]]:
        client = self._client()
        dense, sparse = self._encode_query(question)
        dense_hits = _query_qdrant(client, self.config.collection, "dense", dense, self.config.vector_top_k)
        sparse_hits = _query_qdrant(client, self.config.collection, "sparse", sparse, self.config.vector_top_k)
        return [_article_from_payload(hit) for hit in dense_hits], [_article_from_payload(hit) for hit in sparse_hits]

    def _encode_query(self, question: str) -> tuple[list[float], Any]:
        encoded = _encode_bge_m3(self._embedder_model(), [question])[0]
        sparse = _sparse_vector(encoded["sparse_indices"], encoded["sparse_values"])
        return encoded["dense"], sparse

    def _rerank(self, question: str, candidates: list[ArticleNode]) -> list[ArticleNode]:
        if not candidates:
            return []
        reranker = self._reranker_model()
        pairs = [[question, _article_text(candidate)] for candidate in candidates]
        scores = _compute_rerank_scores(reranker, pairs)
        output: list[ArticleNode] = []
        for score, article in zip(scores, candidates, strict=False):
            clone = ArticleNode.from_dict(article.to_dict())
            clone.score = float(score) + legal_prior_score(clone, question)
            output.append(clone)
        output.sort(key=lambda item: item.score, reverse=True)
        return output

    def _client(self) -> Any:
        if self._qdrant_client is None:
            self._qdrant_client = _new_qdrant_client(self.config.qdrant_url, self.config.qdrant_path)
        return self._qdrant_client

    def _embedder_model(self) -> Any:
        if self._embedder is None:
            self._embedder = _new_embedder(
                self.config.embedding_model,
                self.config.embedding_local_path,
                self.config.local_files_only,
            )
        return self._embedder

    def _reranker_model(self) -> Any:
        if self._reranker is None:
            self._reranker = _new_reranker(
                self.config.reranker_model,
                self.config.reranker_local_path,
                self.config.local_files_only,
            )
        return self._reranker

    def _guidance_hits(self, question: str, plan: LegalQueryPlan) -> list[ArticleNode]:
        guidance_queries = []
        for target in plan.multi_hop_targets[:2]:
            guidance_queries.append(f"{question} {target} hướng dẫn")
        hits: list[ArticleNode] = []
        seen: set[str] = set()
        for query in guidance_queries:
            for article in retrieve_articles(
                self.bm25_index,
                query,
                top_k=max(10, self.config.final_top_k),
                bm25_top_k=max(10, self.config.final_top_k),
                min_score_ratio=0.0,
            ):
                if article.article_key in seen:
                    continue
                seen.add(article.article_key)
                hits.append(article)
        return hits


def build_hybrid_index(
    articles_path: str | Path,
    config: HybridRetrievalConfig,
    *,
    qdrant_client: Any | None = None,
    embedder: Any | None = None,
) -> HybridBuildReport:
    articles = read_articles_jsonl(articles_path)
    if not articles:
        raise HybridRetrievalError("no_articles_to_index")

    index_nodes = _make_index_nodes(articles)
    client = qdrant_client or _new_qdrant_client(config.qdrant_url, config.qdrant_path)
    model = embedder or _new_embedder(config.embedding_model, config.embedding_local_path, config.local_files_only)
    cache_path = _cache_path(index_nodes, config)
    encoded = _load_embedding_cache(cache_path, index_nodes, config.embedding_model)
    if encoded is None:
        encoded = _encode_articles(model, index_nodes, config.batch_size)
        _write_embedding_cache(cache_path, index_nodes, config.embedding_model, encoded)

    _recreate_collection(client, config.collection)
    _upsert_articles(client, config.collection, index_nodes, encoded)
    return HybridBuildReport(
        collection=config.collection,
        articles=len(articles),
        cache_path=str(cache_path),
        embedding_model=config.embedding_model,
        qdrant_url=config.qdrant_url,
        qdrant_path=config.qdrant_path,
        points=len(index_nodes),
    )


def fuse_ranked_lists(ranked_lists: list[list[ArticleNode]], limit: int, k: int = 60) -> list[ArticleNode]:
    scores: dict[str, float] = {}
    articles: dict[str, ArticleNode] = {}
    for ranked in ranked_lists:
        for rank, article in enumerate(ranked, start=1):
            scores[article.article_key] = scores.get(article.article_key, 0.0) + 1.0 / (k + rank)
            if article.article_key not in articles:
                articles[article.article_key] = ArticleNode.from_dict(article.to_dict())

    fused = []
    for key, score in scores.items():
        article = articles[key]
        article.score = score
        fused.append(article)
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused[:limit]


def _with_trace(articles: list[ArticleNode], source: str, query_kind: str) -> list[ArticleNode]:
    output: list[ArticleNode] = []
    for article in articles:
        clone = ArticleNode.from_dict(article.to_dict())
        metadata = dict(clone.metadata)
        existing = str(metadata.get("retrieval_trace") or "")
        trace = f"query_kind={query_kind}; source={source}; score={clone.score:.4f}"
        metadata["retrieval_trace"] = f"{existing} | {trace}" if existing else trace
        clone.metadata = metadata
        output.append(clone)
    return output


def _promote_and_dedupe(articles: list[ArticleNode]) -> list[ArticleNode]:
    output: list[ArticleNode] = []
    seen: set[str] = set()
    for article in articles:
        promoted = _promote_article(article)
        if promoted.article_key in seen:
            continue
        seen.add(promoted.article_key)
        output.append(promoted)
    return output


def _apply_retrieval_bias(articles: list[ArticleNode], plan: LegalQueryPlan) -> list[ArticleNode]:
    adjusted: list[ArticleNode] = []
    for article in articles:
        clone = ArticleNode.from_dict(article.to_dict())
        clone.score = clone.score + _bias_delta(clone, plan)
        adjusted.append(clone)
    adjusted.sort(key=lambda item: item.score, reverse=True)
    return adjusted


def _bias_delta(article: ArticleNode, plan: LegalQueryPlan) -> float:
    title = f"{article.article_title} {article.text[:240]}".lower()
    delta = 0.0
    if plan.retrieval_bias == "content_articles":
        if any(term in title for term in ["trách nhiệm", "thi hành", "hiệu lực", "tổ chức thực hiện", "áp dụng pháp luật"]):
            delta -= 0.25
        if any(facet.lower() in title for facet in plan.legal_facets[:4]):
            delta += 0.15
    elif plan.retrieval_bias == "procedure_articles":
        if any(term in title for term in ["hồ sơ", "thủ tục", "trình tự", "thời hạn", "cơ quan"]):
            delta += 0.2
    elif plan.retrieval_bias == "sanction_articles":
        if any(term in title for term in ["xử phạt", "mức phạt", "khắc phục hậu quả", "vi phạm"]):
            delta += 0.2
    elif plan.retrieval_bias == "authority_articles":
        if any(term in title for term in ["thẩm quyền", "trách nhiệm", "ủy ban", "bộ", "chính phủ"]):
            delta += 0.2
    return delta


def _promote_article(article: ArticleNode) -> ArticleNode:
    parent_key = article.metadata.get("parent_article_key")
    if not parent_key:
        return article
    metadata = dict(article.metadata)
    support_snippet = article.text
    promoted = ArticleNode(
        article_key=str(parent_key),
        doc_id=article.doc_id,
        doc_type=article.doc_type,
        title_for_submission=article.title_for_submission,
        article_label=article.article_label,
        article_title=str(metadata.get("parent_article_title") or article.article_title),
        text=str(metadata.get("parent_text") or article.text),
        status=article.status,
        source_url=article.source_url,
        score=article.score,
        metadata=metadata,
    )
    promoted.metadata["support_snippet"] = support_snippet
    return promoted


def _make_index_nodes(articles: list[ArticleNode]) -> list[ArticleNode]:
    nodes: list[ArticleNode] = []
    for article in articles:
        nodes.append(article)
        nodes.extend(_make_micro_chunks(article))
    return nodes


def _make_micro_chunks(article: ArticleNode, chunk_chars: int = 1400, overlap_chars: int = 240) -> list[ArticleNode]:
    text = article.text.strip()
    if len(text) <= chunk_chars * 2:
        return []
    chunks: list[ArticleNode] = []
    start = 0
    idx = 1
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            split_at = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if split_at > start + chunk_chars // 2:
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            metadata = dict(article.metadata)
            metadata.update(
                {
                    "chunk_type": "micro_chunk",
                    "parent_article_key": article.article_key,
                    "parent_article_title": article.article_title,
                    "parent_text": article.text,
                    "support_snippet": chunk,
                    "chunk_index": idx,
                }
            )
            chunks.append(
                ArticleNode(
                    article_key=f"{article.article_key}::chunk{idx}",
                    doc_id=article.doc_id,
                    doc_type=article.doc_type,
                    title_for_submission=article.title_for_submission,
                    article_label=article.article_label,
                    article_title=article.article_title,
                    text=chunk,
                    status=article.status,
                    source_url=article.source_url,
                    metadata=metadata,
                )
            )
            idx += 1
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def config_from_mapping(mapping: dict[str, Any]) -> HybridRetrievalConfig:
    retrieval = mapping.get("retrieval", {})
    embedding = mapping.get("embedding", {})
    reranker = mapping.get("reranker", {})
    qdrant = mapping.get("qdrant", {})
    return HybridRetrievalConfig(
        qdrant_url=qdrant.get("url", DEFAULT_QDRANT_URL),
        qdrant_path=qdrant.get("path", ""),
        collection=qdrant.get("collection", DEFAULT_COLLECTION),
        embedding_model=embedding.get("model", DEFAULT_EMBEDDING_MODEL),
        reranker_model=reranker.get("model", DEFAULT_RERANKER_MODEL),
        embedding_local_path=embedding.get("local_path", ""),
        reranker_local_path=reranker.get("local_path", ""),
        local_files_only=bool(mapping.get("local_files_only", False) or embedding.get("local_files_only", False) or reranker.get("local_files_only", False)),
        embedding_cache_dir=embedding.get("cache_dir", "data/indices/embedding_cache"),
        bm25_top_k=int(retrieval.get("bm25_top_k", 80)),
        vector_top_k=int(retrieval.get("vector_top_k", 80)),
        fusion_top_k=int(retrieval.get("fusion_top_k", 50)),
        rerank_top_k=int(retrieval.get("rerank_top_k", 30)),
        final_top_k=int(retrieval.get("final_top_k", 5)),
        min_rerank_score=retrieval.get("min_rerank_score"),
        batch_size=int(embedding.get("batch_size", 16)),
    )


def _new_qdrant_client(url: str, path: str = "") -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise HybridRetrievalError("missing_dependency:qdrant-client; install project optional dependency 'rag'") from exc
    if path:
        return QdrantClient(path=path)
    return QdrantClient(url=url)


def _new_embedder(model_name: str, local_path: str = "", local_files_only: bool = False) -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise HybridRetrievalError("missing_dependency:FlagEmbedding; install project optional dependency 'rag'") from exc
    target = local_path or model_name
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        return BGEM3FlagModel(target, use_fp16=True)
    except Exception as exc:  # noqa: BLE001
        if local_files_only:
            raise HybridRetrievalError(f"missing_local_embedding_model:{target}") from exc
        raise


def _new_reranker(model_name: str, local_path: str = "", local_files_only: bool = False) -> Any:
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise HybridRetrievalError("missing_dependency:FlagEmbedding; install project optional dependency 'rag'") from exc
    target = local_path or model_name
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        return FlagReranker(target, use_fp16=True)
    except Exception as exc:  # noqa: BLE001
        if local_files_only:
            raise HybridRetrievalError(f"missing_local_reranker_model:{target}") from exc
        raise


def _encode_articles(model: Any, articles: list[ArticleNode], batch_size: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    texts = [_article_text(article) for article in articles]
    for start in range(0, len(texts), batch_size):
        output.extend(_encode_bge_m3(model, texts[start : start + batch_size]))
    return output


def _encode_bge_m3(model: Any, texts: list[str]) -> list[dict[str, Any]]:
    raw = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense_vecs = raw.get("dense_vecs", [])
    lexical_weights = raw.get("lexical_weights", [])
    encoded = []
    for dense, sparse in zip(dense_vecs, lexical_weights, strict=False):
        indices, values = _normalize_sparse_weights(sparse)
        encoded.append({"dense": _as_float_list(dense), "sparse_indices": indices, "sparse_values": values})
    return encoded


def _compute_rerank_scores(reranker: Any, pairs: list[list[str]]) -> list[float]:
    raw = reranker.compute_score(pairs, normalize=True)
    if isinstance(raw, (float, int)):
        return [float(raw)]
    return [float(score) for score in raw]


def _normalize_sparse_weights(weights: Any) -> tuple[list[int], list[float]]:
    if isinstance(weights, dict):
        items = weights.items()
    else:
        items = []
    indices: list[int] = []
    values: list[float] = []
    for key, value in items:
        try:
            index = int(key)
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score:
            indices.append(index)
            values.append(score)
    return indices, values


def _as_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _sparse_vector(indices: list[int], values: list[float]) -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:
        return {"indices": indices, "values": values}
    return models.SparseVector(indices=indices, values=values)


def _recreate_collection(client: Any, collection: str) -> None:
    try:
        from qdrant_client import models
    except ImportError:
        models = None

    if client.collection_exists(collection):
        client.delete_collection(collection)
    if models is None:
        client.create_collection(
            collection,
            vectors_config={"dense": {"size": 1024, "distance": "Cosine"}},
            sparse_vectors_config={"sparse": {"modifier": "idf"}},
        )
        return
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def _upsert_articles(client: Any, collection: str, articles: list[ArticleNode], encoded: list[dict[str, Any]]) -> None:
    try:
        from qdrant_client import models
    except ImportError:
        models = None

    points = []
    for article, vector in zip(articles, encoded, strict=True):
        if models is None:
            points.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, article.article_key)),
                    "vector": {
                        "dense": vector["dense"],
                        "sparse": {"indices": vector["sparse_indices"], "values": vector["sparse_values"]},
                    },
                    "payload": _article_payload(article),
                }
            )
            continue
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, article.article_key)),
                vector={
                    "dense": vector["dense"],
                    "sparse": models.SparseVector(indices=vector["sparse_indices"], values=vector["sparse_values"]),
                },
                payload=_article_payload(article),
            )
        )
    client.upsert(collection_name=collection, points=points)


def _query_qdrant(client: Any, collection: str, vector_name: str, query: Any, limit: int) -> list[Any]:
    result = client.query_points(
        collection_name=collection,
        query=query,
        using=vector_name,
        with_payload=True,
        limit=limit,
    )
    return list(getattr(result, "points", result))


def _article_payload(article: ArticleNode) -> dict[str, Any]:
    metadata = dict(article.metadata)
    metadata.setdefault("doc_id", article.doc_id)
    metadata.setdefault("article_label", article.article_label)
    metadata.setdefault("law_title", article.title_for_submission)
    metadata.setdefault("text_hash", _text_hash(article.text))
    return {
        "article_key": article.article_key,
        "doc_id": article.doc_id,
        "doc_type": article.doc_type,
        "title_for_submission": article.title_for_submission,
        "article_label": article.article_label,
        "article_title": article.article_title,
        "text": article.text,
        "status": article.status,
        "source_url": article.source_url,
        "metadata": metadata,
    }


def _article_from_payload(hit: Any) -> ArticleNode:
    if isinstance(hit, dict):
        payload = hit.get("payload", {})
        score = float(hit.get("score", 0.0))
    else:
        payload = getattr(hit, "payload", {})
        score = float(getattr(hit, "score", 0.0))
    article = ArticleNode.from_dict(
        {
            "article_key": payload["article_key"],
            "doc_id": payload["doc_id"],
            "doc_type": payload.get("doc_type", ""),
            "title_for_submission": payload["title_for_submission"],
            "article_label": payload["article_label"],
            "article_title": payload.get("article_title", ""),
            "text": payload["text"],
            "status": payload.get("status", ""),
            "source_url": payload.get("source_url", ""),
            "score": score,
            "metadata": payload.get("metadata", {}),
        }
    )
    return article


def _article_text(article: ArticleNode) -> str:
    return "\n".join(
        [
            article.relevant_article,
            article.article_title,
            article.text,
        ]
    )


def _cache_path(articles: list[ArticleNode], config: HybridRetrievalConfig) -> Path:
    digest = hashlib.sha256()
    digest.update(config.embedding_model.encode("utf-8"))
    for article in articles:
        digest.update(article.article_key.encode("utf-8"))
        digest.update(_text_hash(article.text).encode("utf-8"))
    return Path(config.embedding_cache_dir) / f"{digest.hexdigest()[:24]}.jsonl"


def _load_embedding_cache(
    path: Path,
    articles: list[ArticleNode],
    embedding_model: str,
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(articles):
        return None
    encoded: list[dict[str, Any]] = []
    for row, article in zip(rows, articles, strict=True):
        if row.get("embedding_model") != embedding_model:
            return None
        if row.get("article_key") != article.article_key or row.get("text_hash") != _text_hash(article.text):
            return None
        encoded.append(
            {
                "dense": row["dense"],
                "sparse_indices": row["sparse_indices"],
                "sparse_values": row["sparse_values"],
            }
        )
    return encoded


def _write_embedding_cache(
    path: Path,
    articles: list[ArticleNode],
    embedding_model: str,
    encoded: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for article, vector in zip(articles, encoded, strict=True):
            row = {
                "embedding_model": embedding_model,
                "article_key": article.article_key,
                "text_hash": _text_hash(article.text),
                **vector,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_retrieval(
    questions: Iterable[dict[str, Any]],
    expected: dict[int, set[str]],
    search_fn: Any,
    top_k: int,
) -> dict[str, Any]:
    total = 0
    hits = 0
    rows = []
    for item in questions:
        qid = int(item["id"])
        wanted = expected.get(qid, set())
        if not wanted:
            continue
        total += 1
        found = {article.relevant_article for article in search_fn(str(item["question"]), top_k)}
        ok = bool(found & wanted)
        hits += int(ok)
        rows.append({"id": qid, "ok": ok, "expected": sorted(wanted), "found": sorted(found)})
    return {"questions": total, "hit_at_k": hits / total if total else 0.0, "rows": rows}
