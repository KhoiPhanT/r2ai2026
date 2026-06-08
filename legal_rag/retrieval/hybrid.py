from __future__ import annotations

import hashlib
import gc
import json
import math
import os
import uuid
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterable, Iterator

from legal_rag.corpus.ingest import iter_articles_jsonl, read_articles_jsonl
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
    enable_margin_cutoff: bool = True
    margin_delta: float = 0.18
    min_articles_after_cutoff: int = 2
    max_articles_after_cutoff: int = 6
    graph_expand_top_k: int = 24
    batch_size: int = 16
    index_node_types: tuple[str, ...] = ("article", "clause")
    enable_micro_chunks: bool = False
    max_encode_chars: int = 12000
    max_payload_text_chars: int = 20000
    upsert_batch_size: int = 512
    load_bm25: bool = True


@dataclass(slots=True)
class HybridBuildReport:
    collection: str
    articles: int
    cache_path: str
    embedding_model: str
    qdrant_url: str
    qdrant_path: str = ""
    points: int = 0
    clauses: int = 0
    points_level: int = 0
    micro_chunks: int = 0
    graph_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "articles": self.articles,
            "cache_path": self.cache_path,
            "embedding_model": self.embedding_model,
            "qdrant_url": self.qdrant_url,
            "qdrant_path": self.qdrant_path,
            "points": self.points,
            "clauses": self.clauses,
            "points_level": self.points_level,
            "micro_chunks": self.micro_chunks,
            "graph_edges": self.graph_edges,
        }


class HybridRetriever:
    def __init__(
        self,
        articles: list[ArticleNode],
        bm25_index: BM25Index | None,
        config: HybridRetrievalConfig,
        *,
        qdrant_client: Any | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.articles = articles
        self.bm25_index = bm25_index
        self.config = config
        self.articles_by_doc_id = _group_articles_by_doc_id(articles)
        self.reverse_guides_by_doc_id = _reverse_guides_by_doc_id(articles)
        self._last_graph_doc_ids: list[str] = []
        self._qdrant_client = qdrant_client
        self._embedder = embedder
        self._reranker = reranker
        self.last_trace: dict[str, Any] = {}

    @classmethod
    def load(cls, articles_path: str | Path, bm25_index_path: str | Path, config: HybridRetrievalConfig) -> "HybridRetriever":
        articles = read_articles_jsonl(articles_path) if config.load_bm25 else []
        bm25_index = BM25Index.load(bm25_index_path) if config.load_bm25 else None
        return cls(articles, bm25_index, config)

    def search(self, question: str, top_k: int | None = None) -> list[ArticleNode]:
        final_top_k = top_k or self.config.final_top_k
        candidate_lists: list[list[ArticleNode]] = []

        exact_hits = exact_match(self.articles, question) if self.articles else []
        if exact_hits:
            candidate_lists.append(exact_hits)

        bm25_hits = []
        if self.bm25_index is not None and self.config.bm25_top_k > 0:
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
            "actual_backend": "hybrid_qdrant",
        }
        return final_hits

    def search_with_plan(self, question: str, plan: LegalQueryPlan, top_k: int | None = None) -> list[ArticleNode]:
        final_top_k = top_k or self.config.final_top_k
        candidate_lists: list[list[ArticleNode]] = []
        candidate_counts: dict[str, int] = {}
        exact_text = " ".join(
            [question, plan.normalized_question, *plan.target_doc_ids, *plan.target_doc_aliases, *plan.target_article_labels]
        )
        exact_hits = _with_trace(exact_match(self.articles, exact_text), "exact", "planner_targets") if self.articles else []
        if exact_hits:
            candidate_lists.append(exact_hits)
        candidate_counts["exact"] = len(exact_hits)

        for planned_query in plan.queries:
            query = planned_query.text
            bm25_hits = []
            if self.bm25_index is not None and self.config.bm25_top_k > 0:
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

        graph_hits: list[ArticleNode] = []
        seed_hits = fused_seed = fuse_ranked_lists(candidate_lists, limit=max(self.config.fusion_top_k, self.config.rerank_top_k, final_top_k))
        if plan.needs_guidance_docs or plan.target_doc_ids:
            graph_hits = self._graph_hits(question, plan, fused_seed[: max(5, final_top_k)])
            if graph_hits:
                candidate_lists.append(_with_trace(graph_hits, "graph_multihop", "graph"))
            candidate_counts["graph"] = len(graph_hits)
        guidance_hits: list[ArticleNode] = []
        if plan.needs_guidance_docs and not graph_hits:
            guidance_hits = self._guidance_hits(question, plan)
            if guidance_hits:
                candidate_lists.append(_with_trace(guidance_hits, "bm25_guidance", "guidance_fallback"))
            candidate_counts["guidance"] = len(guidance_hits)

        fused = fuse_ranked_lists(candidate_lists, limit=max(self.config.fusion_top_k, self.config.rerank_top_k, final_top_k))
        if not fused:
            self.last_trace = {"candidate_counts": candidate_counts, "fused": 0, "reranked": 0, "final": 0}
            return []
        rerank_query = " ".join(
            [question, plan.normalized_question, " ".join(plan.legal_terms), " ".join(plan.governing_doc_hints[:3])]
        )
        reranked = self._rerank(rerank_query, fused[: self.config.rerank_top_k])
        reranked = _apply_retrieval_bias(reranked, plan)
        reranked = _promote_and_dedupe(reranked)
        if self.config.min_rerank_score is not None:
            reranked = [hit for hit in reranked if hit.score >= self.config.min_rerank_score]
        cutoff_reason = ""
        if self.config.enable_margin_cutoff:
            reranked, cutoff_reason = _margin_cutoff(reranked, self.config)
        final_hits = reranked[: min(final_top_k, self.config.max_articles_after_cutoff)]
        self.last_trace = {
            "candidate_counts": candidate_counts,
            "fused": len(fused),
            "reranked": len(reranked),
            "final": len(final_hits),
            "graph_expansions": self._last_graph_doc_ids or sorted({article.doc_id for article in graph_hits}),
            "threshold_cutoff_reason": cutoff_reason,
            "actual_backend": "hybrid_qdrant",
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
        pairs = [[question, _article_text(candidate, max_chars=self.config.max_encode_chars)] for candidate in candidates]
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
        if self.bm25_index is None:
            return []
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

    def _graph_hits(self, question: str, plan: LegalQueryPlan, seed_hits: list[ArticleNode]) -> list[ArticleNode]:
        if self.bm25_index is None:
            self._last_graph_doc_ids = []
            return []
        related_doc_ids: set[str] = set()
        for article in seed_hits:
            relations = article.metadata.get("document_relations", {})
            related_doc_ids.update(str(item) for item in relations.get("guides_doc_ids", []))
            related_doc_ids.update(str(item) for item in relations.get("amends_doc_ids", []))
            related_doc_ids.update(self.reverse_guides_by_doc_id.get(article.doc_id, []))
        related_doc_ids.difference_update({article.doc_id for article in seed_hits})
        self._last_graph_doc_ids = sorted(related_doc_ids)
        if not related_doc_ids:
            return []
        query = " ".join(
            [
                question,
                *plan.requested_components[:3],
                *plan.target_norm_roles[:2],
                *plan.legal_terms[:3],
                *plan.governing_doc_hints[:2],
            ]
        ).strip()
        hits = retrieve_articles(
            self.bm25_index,
            query or question,
            top_k=self.config.graph_expand_top_k,
            bm25_top_k=max(self.config.graph_expand_top_k * 2, self.config.bm25_top_k),
            min_score_ratio=0.0,
        )
        filtered = []
        seen: set[str] = set()
        for hit in hits:
            if hit.doc_id not in related_doc_ids or hit.article_key in seen:
                continue
            seen.add(hit.article_key)
            filtered.append(hit)
        return filtered


def build_hybrid_index(
    articles_path: str | Path,
    config: HybridRetrievalConfig,
    *,
    qdrant_client: Any | None = None,
    embedder: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> HybridBuildReport:
    _progress(progress, f"scanning articles from {articles_path}")
    preflight = _scan_index_nodes(articles_path, config)
    if not preflight["articles"]:
        raise HybridRetrievalError("no_articles_to_index")

    _progress(
        progress,
        f"expanded {preflight['points']} nodes ({preflight['article_nodes']} articles, "
        f"{preflight['clauses']} clauses, {preflight['points_level']} points)",
    )
    cache_path = _cache_path_from_digest(preflight["digest"], config)
    _progress(progress, f"checking embedding cache {cache_path}")
    cache_prefix = _embedding_cache_valid_prefix_stream(cache_path, articles_path, config, int(preflight["points"]))
    if cache_path.exists() and cache_prefix < int(preflight["points"]):
        _truncate_embedding_cache(cache_path, cache_prefix)
    if cache_prefix < int(preflight["points"]):
        _progress(progress, f"loading embedder {config.embedding_model}")
        model = embedder or _new_embedder(config.embedding_model, config.embedding_local_path, config.local_files_only)
        missing = int(preflight["points"]) - cache_prefix
        _progress(progress, f"encoding {missing} missing nodes to cache in batches of {config.batch_size}")
        _encode_missing_to_cache(
            model,
            articles_path,
            cache_path,
            config,
            start_index=cache_prefix,
            total_nodes=int(preflight["points"]),
            progress=progress,
        )
        del model
        gc.collect()
    else:
        _progress(progress, f"complete cache hit; reused {cache_prefix} nodes without loading embedder")

    _progress(progress, f"opening qdrant store at {config.qdrant_path or config.qdrant_url or 'local'}")
    client = qdrant_client or _new_qdrant_client(config.qdrant_url, config.qdrant_path)
    _progress(progress, f"recreating qdrant collection {config.collection}")
    _recreate_collection(client, config.collection)
    _progress(progress, f"upserting {preflight['points']} cached nodes into qdrant")
    _upsert_cached_embeddings(
        client,
        config.collection,
        articles_path,
        cache_path,
        int(preflight["points"]),
        config,
        progress=progress,
    )
    _progress(progress, "hybrid index build complete")
    return HybridBuildReport(
        collection=config.collection,
        articles=int(preflight["articles"]),
        cache_path=str(cache_path),
        embedding_model=config.embedding_model,
        qdrant_url=config.qdrant_url,
        qdrant_path=config.qdrant_path,
        points=int(preflight["points"]),
        clauses=int(preflight["clauses"]),
        points_level=int(preflight["points_level"]),
        micro_chunks=int(preflight["micro_chunks"]),
        graph_edges=int(preflight["graph_edges"]),
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
    title = f"{article.title_for_submission} {article.article_title} {article.text[:400]}".lower()
    delta = 0.0
    norm_roles = {str(role).lower() for role in article.metadata.get("norm_roles", [])}
    node_type = str(article.metadata.get("node_type") or article.metadata.get("chunk_type") or "article").lower()
    must_terms = [term.lower() for term in plan.filters.get("must_include_terms", []) if term]
    should_terms = [term.lower() for term in plan.filters.get("should_include_terms", []) if term]
    matched_must = sum(1 for term in must_terms if term in title)
    matched_should = sum(1 for term in should_terms if term in title)
    governing_hints = [hint.lower() for hint in plan.governing_doc_hints if hint]
    matched_governing = sum(1 for hint in governing_hints if hint in title)
    anchor_map = {
        "support_policy_sme": ("hỗ trợ doanh nghiệp nhỏ và vừa", "doanh nghiệp nhỏ và vừa", "cơ sở ươm tạo", "khu làm việc chung"),
        "labor_sanctions": ("lao động", "người lao động", "hợp đồng lao động", "bằng cấp", "văn bằng", "chứng chỉ"),
        "tax_admin_penalties": ("thuế", "hóa đơn", "quản lý thuế", "mã số thuế"),
        "intellectual_property": ("sở hữu trí tuệ", "nhãn hiệu", "sáng chế", "kiểu dáng"),
    }
    matched_anchors = 0
    for anchor in plan.domain_anchors:
        terms = anchor_map.get(anchor, ())
        if any(term in title for term in terms):
            matched_anchors += 1

    if must_terms:
        if matched_must == 0:
            delta -= 1.0
        else:
            delta += 0.2 * matched_must
    if should_terms and matched_should:
        delta += min(0.3, 0.08 * matched_should)
    if governing_hints:
        if matched_governing:
            delta += min(0.45, 0.16 * matched_governing)
        else:
            delta -= 0.35
    if plan.domain_anchors:
        if matched_anchors:
            delta += min(0.55, 0.22 * matched_anchors)
        else:
            delta -= 0.45
    if node_type in {"clause", "point"}:
        delta += 0.12
    if plan.target_norm_roles:
        matched_roles = sum(1 for role in plan.target_norm_roles if role.lower() in norm_roles)
        if matched_roles:
            delta += 0.22 * matched_roles
        elif norm_roles and plan.retrieval_bias in {"procedure_articles", "sanction_articles", "authority_articles"}:
            delta -= 0.25

    if plan.retrieval_bias == "content_articles":
        if any(term in title for term in ["trách nhiệm", "thi hành", "hiệu lực", "tổ chức thực hiện", "áp dụng pháp luật"]):
            delta -= 0.4
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
    if plan.needs_guidance_docs:
        if article.doc_type.lower() in {"nghị định", "thông tư"}:
            delta += 0.15
        elif article.doc_type.lower() == "luật":
            delta -= 0.05
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
    if metadata.get("support_span_key"):
        promoted.metadata["support_span_key"] = metadata.get("support_span_key")
    if metadata.get("support_span_label"):
        promoted.metadata["support_span_label"] = metadata.get("support_span_label")
    return promoted


def _make_index_nodes(
    articles: list[ArticleNode],
    index_node_types: tuple[str, ...] = ("article", "clause"),
    enable_micro_chunks: bool = False,
) -> list[ArticleNode]:
    node_types = {node_type.lower() for node_type in index_node_types}
    nodes: list[ArticleNode] = []
    for article in articles:
        if "article" in node_types:
            nodes.append(article)
        span_nodes = _make_span_nodes(article, node_types)
        nodes.extend(span_nodes)
        if enable_micro_chunks:
            for node in span_nodes:
                nodes.extend(_make_micro_chunks(node))
        if enable_micro_chunks and not span_nodes:
            nodes.extend(_make_micro_chunks(article))
    return nodes


def _iter_index_nodes_from_path(articles_path: str | Path, config: HybridRetrievalConfig) -> Iterator[ArticleNode]:
    for article in iter_articles_jsonl(articles_path):
        yield from _make_index_nodes([article], config.index_node_types, config.enable_micro_chunks)


def _scan_index_nodes(articles_path: str | Path, config: HybridRetrievalConfig) -> dict[str, int | str]:
    digest = hashlib.sha256()
    digest.update(config.embedding_model.encode("utf-8"))
    digest.update(str(config.max_encode_chars).encode("utf-8"))
    counts = {
        "articles": 0,
        "points": 0,
        "article_nodes": 0,
        "clauses": 0,
        "points_level": 0,
        "micro_chunks": 0,
        "graph_edges": 0,
    }
    for article in iter_articles_jsonl(articles_path):
        counts["articles"] += 1
        counts["graph_edges"] += len(article.metadata.get("graph_neighbors", []))
        for node in _make_index_nodes([article], config.index_node_types, config.enable_micro_chunks):
            node_type = str(node.metadata.get("node_type") or node.metadata.get("chunk_type") or "article")
            counts["points"] += 1
            if node_type == "article":
                counts["article_nodes"] += 1
            elif node_type == "clause":
                counts["clauses"] += 1
            elif node_type == "point":
                counts["points_level"] += 1
            elif node_type == "micro_chunk":
                counts["micro_chunks"] += 1
            digest.update(node.article_key.encode("utf-8"))
            digest.update(_embedding_text_hash(node, config).encode("utf-8"))
    counts["digest"] = digest.hexdigest()[:24]
    return counts


def _make_span_nodes(article: ArticleNode, node_types: set[str] | None = None) -> list[ArticleNode]:
    allowed = node_types or {"clause", "point"}
    nodes: list[ArticleNode] = []
    for clause in article.metadata.get("clause_nodes", []):
        clause_text = str(clause.get("text") or "").strip()
        if clause_text and "clause" in allowed:
            nodes.append(_span_node(article, clause, "clause"))
        for point in clause.get("point_nodes", []):
            point_text = str(point.get("text") or "").strip()
            if point_text and "point" in allowed:
                nodes.append(_span_node(article, point, "point"))
    return nodes


def _span_node(article: ArticleNode, span: dict[str, Any], node_type: str) -> ArticleNode:
    metadata = _compact_payload_metadata(article.metadata)
    metadata.update(
        {
            "chunk_type": node_type,
            "node_type": node_type,
            "parent_article_key": article.article_key,
            "parent_article_title": article.article_title,
            "support_snippet": str(span.get("text") or ""),
            "support_span_key": str(span.get("span_key") or ""),
            "support_span_label": str(span.get("label") or ""),
        }
    )
    return ArticleNode(
        article_key=f"{article.article_key}::{node_type}:{span.get('span_key')}",
        doc_id=article.doc_id,
        doc_type=article.doc_type,
        title_for_submission=article.title_for_submission,
        article_label=article.article_label,
        article_title=article.article_title,
        text=str(span.get("text") or ""),
        status=article.status,
        source_url=article.source_url,
        metadata=metadata,
    )


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
                    "node_type": "micro_chunk",
                    "parent_article_key": article.article_key,
                    "parent_article_title": article.article_title,
                    "parent_text": article.text,
                    "support_snippet": chunk,
                    "chunk_index": idx,
                    "support_span_key": article.metadata.get("support_span_key", ""),
                    "support_span_label": article.metadata.get("support_span_label", ""),
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
        enable_margin_cutoff=bool(retrieval.get("enable_margin_cutoff", True)),
        margin_delta=float(retrieval.get("margin_delta", 0.18)),
        min_articles_after_cutoff=int(retrieval.get("min_articles_after_cutoff", 2)),
        max_articles_after_cutoff=int(retrieval.get("max_articles_after_cutoff", 6)),
        graph_expand_top_k=int(retrieval.get("graph_expand_top_k", 24)),
        batch_size=int(embedding.get("batch_size", 16)),
        index_node_types=tuple(str(item).lower() for item in retrieval.get("index_node_types", ["article", "clause"])),
        enable_micro_chunks=bool(retrieval.get("enable_micro_chunks", False)),
        max_encode_chars=int(embedding.get("max_encode_chars", retrieval.get("max_encode_chars", 12000))),
        max_payload_text_chars=int(retrieval.get("max_payload_text_chars", 20000)),
        upsert_batch_size=int(retrieval.get("upsert_batch_size", 512)),
        load_bm25=bool(retrieval.get("load_bm25", True)),
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
    target = _resolve_model_target(model_name, local_path)
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
    target = _resolve_model_target(model_name, local_path)
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        return FlagReranker(target, use_fp16=True)
    except Exception as exc:  # noqa: BLE001
        if local_files_only:
            raise HybridRetrievalError(f"missing_local_reranker_model:{target}") from exc
        raise


def _resolve_model_target(model_name: str, local_path: str) -> str:
    if local_path:
        return local_path
    cached = _cached_snapshot_path(model_name)
    return cached or model_name


def _cached_snapshot_path(model_name: str) -> str:
    if "/" not in model_name:
        return ""
    org, repo = model_name.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}" / "snapshots"
    if not cache_root.exists():
        return ""
    snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
    if not snapshots:
        return ""
    return str(snapshots[-1])


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _encode_articles(
    model: Any,
    articles: list[ArticleNode],
    batch_size: int,
    *,
    max_chars: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total_batches = max(1, (len(articles) + batch_size - 1) // batch_size)
    for batch_index, start in enumerate(range(0, len(articles), batch_size), start=1):
        batch = articles[start : start + batch_size]
        texts = [_article_text(article, max_chars=max_chars) for article in batch]
        output.extend(_encode_bge_m3(model, texts))
        if progress and (batch_index == 1 or batch_index == total_batches or batch_index % 25 == 0):
            progress(f"encoded batch {batch_index}/{total_batches}")
    return output


def _encode_missing_to_cache(
    model: Any,
    articles_path: str | Path,
    cache_path: Path,
    config: HybridRetrievalConfig,
    *,
    start_index: int,
    total_nodes: int,
    progress: Callable[[str], None] | None = None,
) -> None:
    total_batches = max(1, (total_nodes + config.batch_size - 1) // config.batch_size)
    batch: list[ArticleNode] = []
    batch_index = start_index // config.batch_size + 1
    for node_index, article in enumerate(_iter_index_nodes_from_path(articles_path, config)):
        if node_index < start_index:
            continue
        batch.append(article)
        if len(batch) < config.batch_size:
            continue
        _encode_cache_batch(model, batch, cache_path, config)
        if progress and (batch_index == 1 or batch_index == total_batches or batch_index % 25 == 0):
            progress(f"encoded+cached batch {batch_index}/{total_batches}")
        batch = []
        batch_index += 1
    if batch:
        _encode_cache_batch(model, batch, cache_path, config)
        if progress and (batch_index == 1 or batch_index == total_batches or batch_index % 25 == 0):
            progress(f"encoded+cached batch {batch_index}/{total_batches}")


def _encode_cache_batch(
    model: Any,
    batch: list[ArticleNode],
    cache_path: Path,
    config: HybridRetrievalConfig,
) -> None:
    texts = [_article_text(article, max_chars=config.max_encode_chars) for article in batch]
    vectors = _encode_bge_m3(model, texts)
    _append_embedding_cache(cache_path, batch, config, vectors)


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


def _upsert_articles(
    client: Any,
    collection: str,
    articles: list[ArticleNode],
    encoded: list[dict[str, Any]],
    *,
    config: HybridRetrievalConfig,
    progress: Callable[[str], None] | None = None,
) -> None:
    _upsert_article_vectors(
        client,
        collection,
        ((index, article, vector) for index, (article, vector) in enumerate(zip(articles, encoded, strict=True), start=1)),
        total=len(articles),
        config=config,
        progress=progress,
    )


def _upsert_cached_embeddings(
    client: Any,
    collection: str,
    articles_path: str | Path,
    cache_path: Path,
    limit: int,
    config: HybridRetrievalConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    _upsert_article_vectors(
        client,
        collection,
        _iter_cached_article_vectors_stream(cache_path, articles_path, limit, config),
        total=limit,
        config=config,
        progress=progress,
    )


def _upsert_article_vectors(
    client: Any,
    collection: str,
    article_vectors: Iterable[tuple[int, ArticleNode, dict[str, Any]]],
    *,
    total: int,
    config: HybridRetrievalConfig,
    progress: Callable[[str], None] | None = None,
) -> None:
    try:
        from qdrant_client import models
    except ImportError:
        models = None

    batch: list[Any] = []
    upserted = 0
    for index, article, vector in article_vectors:
        if models is None:
            batch.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, article.article_key)),
                    "vector": {
                        "dense": vector["dense"],
                        "sparse": {"indices": vector["sparse_indices"], "values": vector["sparse_values"]},
                    },
                    "payload": _article_payload(article, config.max_payload_text_chars),
                }
            )
        else:
            batch.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, article.article_key)),
                    vector={
                        "dense": vector["dense"],
                        "sparse": models.SparseVector(indices=vector["sparse_indices"], values=vector["sparse_values"]),
                    },
                    payload=_article_payload(article, config.max_payload_text_chars),
                )
            )
        if len(batch) >= config.upsert_batch_size or index == total:
            client.upsert(collection_name=collection, points=batch)
            upserted += len(batch)
            batch = []
        if progress and (index == 1 or index == total or index % 1000 == 0):
            progress(f"prepared {index}/{total} qdrant points")
    if batch:
        client.upsert(collection_name=collection, points=batch)
        upserted += len(batch)
    if progress:
        progress(f"upserted {upserted} qdrant points")


def _query_qdrant(client: Any, collection: str, vector_name: str, query: Any, limit: int) -> list[Any]:
    result = client.query_points(
        collection_name=collection,
        query=query,
        using=vector_name,
        with_payload=True,
        limit=limit,
    )
    return list(getattr(result, "points", result))


def _article_payload(article: ArticleNode, max_text_chars: int) -> dict[str, Any]:
    metadata = _compact_payload_metadata(article.metadata)
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
        "text": _truncate_text(article.text, max_text_chars),
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


def _group_articles_by_doc_id(articles: list[ArticleNode]) -> dict[str, list[ArticleNode]]:
    output: dict[str, list[ArticleNode]] = {}
    for article in articles:
        output.setdefault(article.doc_id, []).append(article)
    return output


def _reverse_guides_by_doc_id(articles: list[ArticleNode]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for article in articles:
        relations = article.metadata.get("document_relations", {})
        for target_doc_id in relations.get("guides_doc_ids", []):
            output.setdefault(str(target_doc_id), set()).add(article.doc_id)
    return output


def _margin_cutoff(articles: list[ArticleNode], config: HybridRetrievalConfig) -> tuple[list[ArticleNode], str]:
    if not articles:
        return [], ""
    kept = [articles[0]]
    cutoff_reason = ""
    for previous, current in zip(articles, articles[1:], strict=False):
        if len(kept) >= config.max_articles_after_cutoff:
            cutoff_reason = f"max_articles:{config.max_articles_after_cutoff}"
            break
        margin = previous.score - current.score
        if len(kept) >= config.min_articles_after_cutoff and margin > config.margin_delta:
            cutoff_reason = f"margin_drop:{margin:.4f}"
            break
        kept.append(current)
    if not cutoff_reason and len(kept) < len(articles):
        cutoff_reason = "full_rank_retained"
    return kept, cutoff_reason


def _article_text(article: ArticleNode, max_chars: int | None = None) -> str:
    text = "\n".join(
        [
            article.relevant_article,
            article.article_title,
            article.text,
        ]
    )
    return _truncate_text(text, max_chars) if max_chars else text


def _truncate_text(text: str, max_chars: int | None) -> str:
    if not max_chars or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _compact_payload_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "chunk_type",
        "node_type",
        "doc_id",
        "article_label",
        "article_no",
        "article_order",
        "law_title",
        "part_label",
        "part_title",
        "chapter_label",
        "chapter_title",
        "section_label",
        "section_title",
        "source_dataset",
        "source_document_id",
        "html_url",
        "expiration_date",
        "language",
        "text_hash",
        "document_relations",
        "graph_neighbors",
        "norm_roles",
        "primary_norm_role",
        "retrieval_trace",
        "parent_article_key",
        "parent_article_title",
        "support_snippet",
        "support_span_key",
        "support_span_label",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def _cache_path(articles: list[ArticleNode], config: HybridRetrievalConfig) -> Path:
    digest = hashlib.sha256()
    digest.update(config.embedding_model.encode("utf-8"))
    digest.update(str(config.max_encode_chars).encode("utf-8"))
    for article in articles:
        digest.update(article.article_key.encode("utf-8"))
        digest.update(_embedding_text_hash(article, config).encode("utf-8"))
    return Path(config.embedding_cache_dir) / f"{digest.hexdigest()[:24]}.jsonl"


def _cache_path_from_digest(digest: str | int, config: HybridRetrievalConfig) -> Path:
    return Path(config.embedding_cache_dir) / f"{digest}.jsonl"


def _embedding_cache_valid_prefix(
    path: Path,
    articles: list[ArticleNode],
    config: HybridRetrievalConfig,
) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if count >= len(articles):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return count
            if not _cache_row_matches(row, articles[count], config):
                return count
            count += 1
    return count


def _embedding_cache_valid_prefix_stream(
    path: Path,
    articles_path: str | Path,
    config: HybridRetrievalConfig,
    total_nodes: int,
) -> int:
    if not path.exists():
        return 0
    count = 0
    nodes = _iter_index_nodes_from_path(articles_path, config)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if count >= total_nodes:
                break
            try:
                row = json.loads(line)
                article = next(nodes)
            except (json.JSONDecodeError, StopIteration):
                return count
            if not _cache_row_matches(row, article, config):
                return count
            count += 1
    return count


def _iter_cached_article_vectors(
    path: Path,
    articles: list[ArticleNode],
    limit: int,
    config: HybridRetrievalConfig,
) -> Iterator[tuple[int, ArticleNode, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if index > limit:
                break
            row = json.loads(line)
            article = articles[index - 1]
            if not _cache_row_matches(row, article, config):
                raise HybridRetrievalError(f"invalid_embedding_cache_row:{path}:{index}")
            yield index, article, _vector_from_cache_row(row)


def _iter_cached_article_vectors_stream(
    path: Path,
    articles_path: str | Path,
    limit: int,
    config: HybridRetrievalConfig,
) -> Iterator[tuple[int, ArticleNode, dict[str, Any]]]:
    nodes = _iter_index_nodes_from_path(articles_path, config)
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if index > limit:
                break
            row = json.loads(line)
            try:
                article = next(nodes)
            except StopIteration as exc:
                raise HybridRetrievalError(f"embedding_cache_longer_than_nodes:{path}:{index}") from exc
            if not _cache_row_matches(row, article, config):
                raise HybridRetrievalError(f"invalid_embedding_cache_row:{path}:{index}")
            yield index, article, _vector_from_cache_row(row)


def _truncate_embedding_cache(path: Path, valid_prefix: int) -> None:
    if valid_prefix <= 0:
        path.unlink(missing_ok=True)
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src, start=1):
            if index > valid_prefix:
                break
            dst.write(line)
    tmp_path.replace(path)


def _append_embedding_cache(
    path: Path,
    articles: list[ArticleNode],
    config: HybridRetrievalConfig,
    encoded: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for article, vector in zip(articles, encoded, strict=True):
            row = {
                "embedding_model": config.embedding_model,
                "article_key": article.article_key,
                "text_hash": _embedding_text_hash(article, config),
                "max_encode_chars": config.max_encode_chars,
                **vector,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cache_row_matches(row: dict[str, Any], article: ArticleNode, config: HybridRetrievalConfig) -> bool:
    return (
        row.get("embedding_model") == config.embedding_model
        and row.get("article_key") == article.article_key
        and row.get("text_hash") == _embedding_text_hash(article, config)
        and int(row.get("max_encode_chars") or 0) == config.max_encode_chars
        and isinstance(row.get("dense"), list)
        and isinstance(row.get("sparse_indices"), list)
        and isinstance(row.get("sparse_values"), list)
    )


def _vector_from_cache_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dense": row["dense"],
        "sparse_indices": row["sparse_indices"],
        "sparse_values": row["sparse_values"],
    }


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


def _embedding_text_hash(article: ArticleNode, config: HybridRetrievalConfig) -> str:
    return _text_hash(_article_text(article, max_chars=config.max_encode_chars))


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
