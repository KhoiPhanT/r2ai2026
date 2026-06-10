from __future__ import annotations

import gzip
import hashlib
import gc
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from legal_rag.corpus.ingest import iter_articles_jsonl, read_articles_jsonl
from legal_rag.domain.components import COMPONENT_REGISTRY, component_matches
from legal_rag.lexicon import normalize_legal_query_text
from legal_rag.planner import LegalQueryPlan
from legal_rag.retrieval.bm25 import BM25Index
from legal_rag.retrieval.fts import FTS5Index
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
    rerank_max_chars: int = 2800
    max_payload_text_chars: int = 20000
    upsert_batch_size: int = 512
    load_bm25: bool = True
    lexical_index_path: str = ""
    max_embedded_points: int = 20_000


@dataclass(slots=True)
class RetrievalBudget:
    bm25_top_k: int
    vector_top_k: int
    fusion_top_k: int
    rerank_top_k: int
    final_top_k: int
    graph_expand_top_k: int
    margin_delta: float
    min_articles_after_cutoff: int
    max_articles_after_cutoff: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bm25_top_k": self.bm25_top_k,
            "vector_top_k": self.vector_top_k,
            "fusion_top_k": self.fusion_top_k,
            "rerank_top_k": self.rerank_top_k,
            "final_top_k": self.final_top_k,
            "graph_expand_top_k": self.graph_expand_top_k,
            "margin_delta": self.margin_delta,
            "min_articles_after_cutoff": self.min_articles_after_cutoff,
            "max_articles_after_cutoff": self.max_articles_after_cutoff,
        }


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
        lexical_index: Any | None = None,
    ) -> None:
        self.articles = articles
        self.bm25_index = bm25_index
        self.lexical_index = lexical_index or bm25_index
        self.config = config
        self.article_by_key = {article.article_key: article for article in articles}
        self.articles_by_doc_id = _group_articles_by_doc_id(articles)
        self.reverse_guides_by_doc_id = _reverse_guides_by_doc_id(articles)
        self._last_graph_doc_ids: list[str] = []
        self._qdrant_client = qdrant_client
        self._embedder = embedder
        self._reranker = reranker
        self.last_trace: dict[str, Any] = {}

    @classmethod
    def load(cls, articles_path: str | Path, bm25_index_path: str | Path, config: HybridRetrievalConfig) -> "HybridRetriever":
        if config.lexical_index_path and Path(config.lexical_index_path).exists():
            lexical_index = FTS5Index.load(config.lexical_index_path)
            return cls([], None, config, lexical_index=lexical_index)
        bm25_index = BM25Index.load(bm25_index_path) if config.load_bm25 else None
        articles = bm25_index.articles if bm25_index is not None else read_articles_jsonl(articles_path) if config.load_bm25 else []
        return cls(articles, bm25_index, config, lexical_index=bm25_index)

    def search(self, question: str, top_k: int | None = None) -> list[ArticleNode]:
        final_top_k = top_k or self.config.final_top_k
        candidate_lists: list[list[ArticleNode]] = []

        exact_hits = self._exact_hits(question)
        if exact_hits:
            candidate_lists.append(exact_hits)

        bm25_hits = []
        if self.lexical_index is not None and self.config.bm25_top_k > 0:
            bm25_hits = self._lexical_search(question, self.config.bm25_top_k)
        if bm25_hits:
            candidate_lists.append(bm25_hits)

        qdrant_start = time.perf_counter()
        dense_hits, sparse_hits = self._search_qdrant(question)
        qdrant_ms = (time.perf_counter() - qdrant_start) * 1000.0
        if dense_hits:
            candidate_lists.append(dense_hits)
        if sparse_hits:
            candidate_lists.append(sparse_hits)

        fused = fuse_ranked_lists(candidate_lists, limit=max(self.config.rerank_top_k, final_top_k))
        if not fused:
            self.last_trace = {"candidate_counts": {}, "fused": 0, "reranked": 0, "final": 0}
            return []

        rerank_start = time.perf_counter()
        reranked = self._rerank(question, fused[: self.config.rerank_top_k])
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        reranked = _promote_and_dedupe(reranked)
        if self.config.min_rerank_score is not None:
            reranked = [hit for hit in reranked if hit.score >= self.config.min_rerank_score]
        if self.config.enable_margin_cutoff:
            reranked, _cutoff_reason = _margin_cutoff(reranked, self.config)
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
            "qdrant_ms": round(qdrant_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
        }
        return final_hits

    def search_with_plan(self, question: str, plan: LegalQueryPlan, top_k: int | None = None) -> list[ArticleNode]:
        requested_top_k = top_k or self.config.final_top_k
        budget = _budget_for_plan(plan, self.config, requested_top_k)
        final_top_k = budget.final_top_k
        planned_queries = _dedupe_planned_queries(plan.queries)
        semantic_query_count = _semantic_query_count(planned_queries)
        bm25_limit = _per_query_limit(budget.bm25_top_k, max(1, len(planned_queries)), floor=max(12, final_top_k * 3))
        vector_limit = _per_query_limit(budget.vector_top_k, max(1, semantic_query_count), floor=max(12, final_top_k * 3))
        candidate_lists: list[list[ArticleNode]] = []
        candidate_counts: dict[str, int] = {}
        query_branches: list[dict[str, Any]] = []
        qdrant_total_ms = 0.0
        exact_text = " ".join(
            [question, plan.normalized_question, *plan.target_doc_ids, *plan.target_doc_aliases, *plan.target_article_labels]
        )
        exact_hits = _with_trace(self._exact_hits(exact_text), "exact", "planner_targets")
        if exact_hits:
            candidate_lists.append(exact_hits)
        candidate_counts["exact"] = len(exact_hits)

        for planned_query in planned_queries:
            query = planned_query.text
            branch = {"kind": planned_query.kind, "text": query, "bm25_limit": bm25_limit, "vector_limit": 0}
            bm25_hits = []
            if self.lexical_index is not None and self.config.bm25_top_k > 0:
                bm25_hits = self._lexical_search(query, bm25_limit)
            if bm25_hits:
                candidate_lists.append(_with_trace(bm25_hits, "bm25", planned_query.kind))
            candidate_counts[f"bm25:{planned_query.kind}"] = len(bm25_hits)
            branch["bm25_hits"] = len(bm25_hits)

            dense_hits: list[ArticleNode] = []
            sparse_hits: list[ArticleNode] = []
            if _query_uses_qdrant(planned_query, semantic_query_count) and vector_limit > 0:
                branch["vector_limit"] = vector_limit
                qdrant_start = time.perf_counter()
                dense_hits, sparse_hits = self._search_qdrant(query, limit=vector_limit)
                qdrant_ms = (time.perf_counter() - qdrant_start) * 1000.0
                qdrant_total_ms += qdrant_ms
                branch["qdrant_ms"] = round(qdrant_ms, 2)
                if dense_hits:
                    candidate_lists.append(_with_trace(dense_hits, "dense", planned_query.kind))
                if sparse_hits:
                    candidate_lists.append(_with_trace(sparse_hits, "sparse", planned_query.kind))
            candidate_counts[f"dense:{planned_query.kind}"] = len(dense_hits)
            candidate_counts[f"sparse:{planned_query.kind}"] = len(sparse_hits)
            branch["dense_hits"] = len(dense_hits)
            branch["sparse_hits"] = len(sparse_hits)
            query_branches.append(branch)

        graph_hits: list[ArticleNode] = []
        seed_hits = fuse_ranked_lists(candidate_lists, limit=max(budget.fusion_top_k, budget.rerank_top_k, final_top_k))
        if plan.needs_guidance_docs or plan.target_doc_ids:
            graph_hits = self._graph_hits(question, plan, seed_hits[: max(5, final_top_k)], limit=budget.graph_expand_top_k)
            if graph_hits:
                candidate_lists.append(_with_trace(graph_hits, "graph_multihop", "graph"))
            candidate_counts["graph"] = len(graph_hits)
        guidance_hits: list[ArticleNode] = []
        if plan.needs_guidance_docs and not graph_hits:
            guidance_hits = self._guidance_hits(question, plan, limit=max(10, budget.graph_expand_top_k))
            if guidance_hits:
                candidate_lists.append(_with_trace(guidance_hits, "bm25_guidance", "guidance_fallback"))
            candidate_counts["guidance"] = len(guidance_hits)

        fused = fuse_ranked_lists(candidate_lists, limit=max(budget.fusion_top_k, budget.rerank_top_k, final_top_k))
        if not fused:
            self.last_trace = {"candidate_counts": candidate_counts, "fused": 0, "reranked": 0, "final": 0}
            return []
        rerank_query = _rerank_query_for_plan(question, plan)
        rerank_start = time.perf_counter()
        reranked = self._rerank(rerank_query, fused[: budget.rerank_top_k])
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        reranked = _promote_and_dedupe(reranked)
        reranked = self._enrich_parent_articles(reranked)
        reranked = _apply_retrieval_bias(reranked, plan)
        reranked, superseded_doc_ids = self._suppress_superseded_candidates(reranked)
        reranked, scope_filtered_doc_ids = _suppress_unrequested_local_scope(reranked, plan, question=question)
        if self.config.min_rerank_score is not None:
            reranked = [hit for hit in reranked if hit.score >= self.config.min_rerank_score]
        cutoff_reason = ""
        if self.config.enable_margin_cutoff:
            reranked, cutoff_reason = _margin_cutoff_with_budget(reranked, budget)
        final_hits = reranked[: min(final_top_k, budget.max_articles_after_cutoff)]
        self.last_trace = {
            "candidate_counts": candidate_counts,
            "fused": len(fused),
            "reranked": len(reranked),
            "final": len(final_hits),
            "retrieval_budget": budget.to_dict(),
            "planned_query_count": len(planned_queries),
            "semantic_query_count": semantic_query_count,
            "query_branches": query_branches,
            "rerank_query": rerank_query,
            "graph_expansions": self._last_graph_doc_ids or sorted({article.doc_id for article in graph_hits}),
            "threshold_cutoff_reason": cutoff_reason,
            "superseded_doc_ids": sorted(superseded_doc_ids),
            "scope_filtered_doc_ids": sorted(scope_filtered_doc_ids),
            "actual_backend": "hybrid_qdrant",
            "qdrant_ms": round(qdrant_total_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
        }
        return final_hits

    def search_targeted_component(
        self,
        question: str,
        plan: LegalQueryPlan,
        component: str,
        current_articles: list[ArticleNode],
        *,
        top_k: int | None = None,
    ) -> list[ArticleNode]:
        """Retrieve one missing required component without replaying the full query plan."""
        final_top_k = top_k or self.config.final_top_k
        query = _targeted_component_query(question, plan, component)
        lexical_limit = min(24, max(12, final_top_k * 4))
        vector_limit = min(20, max(10, final_top_k * 3))
        candidate_lists: list[list[ArticleNode]] = [current_articles]

        lexical_hits = self._lexical_search(query, lexical_limit) if self.lexical_index is not None else []
        if lexical_hits:
            candidate_lists.append(_with_trace(lexical_hits, "fts", f"repair:{component}"))
        qdrant_start = time.perf_counter()
        dense_hits, sparse_hits = self._search_qdrant(query, limit=vector_limit)
        qdrant_ms = (time.perf_counter() - qdrant_start) * 1000.0
        if dense_hits:
            candidate_lists.append(_with_trace(dense_hits, "dense", f"repair:{component}"))
        if sparse_hits:
            candidate_lists.append(_with_trace(sparse_hits, "sparse", f"repair:{component}"))

        rerank_limit = min(self.config.rerank_top_k, max(12, final_top_k * 4))
        fused = fuse_ranked_lists(candidate_lists, limit=rerank_limit)
        rerank_start = time.perf_counter()
        reranked = self._rerank(query, fused)
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        reranked = self._enrich_parent_articles(_promote_and_dedupe(reranked))
        reranked = _apply_retrieval_bias(reranked, plan)
        reranked, superseded_doc_ids = self._suppress_superseded_candidates(reranked)
        reranked, scope_filtered_doc_ids = _suppress_unrequested_local_scope(reranked, plan, question=question)
        final_hits = reranked[:final_top_k]
        previous_trace = dict(self.last_trace)
        self.last_trace = {
            **previous_trace,
            "targeted_repair": component,
            "targeted_repair_query": query,
            "targeted_repair_candidates": {
                "current": len(current_articles),
                "lexical": len(lexical_hits),
                "dense": len(dense_hits),
                "sparse": len(sparse_hits),
            },
            "superseded_doc_ids": sorted(
                set(previous_trace.get("superseded_doc_ids", [])) | superseded_doc_ids
            ),
            "scope_filtered_doc_ids": sorted(
                set(previous_trace.get("scope_filtered_doc_ids", [])) | scope_filtered_doc_ids
            ),
            "qdrant_ms": round(float(previous_trace.get("qdrant_ms", 0.0)) + qdrant_ms, 2),
            "rerank_ms": round(float(previous_trace.get("rerank_ms", 0.0)) + rerank_ms, 2),
            "final": len(final_hits),
        }
        return final_hits

    def _suppress_superseded_candidates(self, articles: list[ArticleNode]) -> tuple[list[ArticleNode], set[str]]:
        if not articles or not hasattr(self.lexical_index, "superseded_doc_ids"):
            return articles, set()
        candidate_doc_ids = {article.doc_id for article in articles if article.doc_id}
        superseded = set(self.lexical_index.superseded_doc_ids(candidate_doc_ids))
        if not superseded:
            return articles, set()
        kept = [article for article in articles if article.doc_id not in superseded]
        return (kept or articles), superseded

    def _enrich_parent_articles(self, articles: list[ArticleNode]) -> list[ArticleNode]:
        output: list[ArticleNode] = []
        for article in articles:
            parent = self.article_by_key.get(article.article_key)
            if parent is None and hasattr(self.lexical_index, "get_article"):
                parent = self.lexical_index.get_article(article.article_key)
            if parent is None or _article_richness(article) >= _article_richness(parent):
                output.append(article)
                continue
            enriched = ArticleNode.from_dict(parent.to_dict())
            enriched.score = article.score
            metadata = dict(enriched.metadata)
            for key in ("retrieval_trace", "support_snippet", "support_span_key", "support_span_label"):
                if article.metadata.get(key):
                    metadata[key] = article.metadata.get(key)
            enriched.metadata = metadata
            output.append(enriched)
        return output

    def _search_qdrant(self, question: str, limit: int | None = None) -> tuple[list[ArticleNode], list[ArticleNode]]:
        client = self._client()
        dense, sparse = self._encode_query(question)
        top_k = limit or self.config.vector_top_k
        dense_hits = _query_qdrant(client, self.config.collection, "dense", dense, top_k)
        sparse_hits = _query_qdrant(client, self.config.collection, "sparse", sparse, top_k)
        return [_article_from_payload(hit) for hit in dense_hits], [_article_from_payload(hit) for hit in sparse_hits]

    def _encode_query(self, question: str) -> tuple[list[float], Any]:
        encoded = _encode_bge_m3(self._embedder_model(), [question])[0]
        sparse = _sparse_vector(encoded["sparse_indices"], encoded["sparse_values"])
        return encoded["dense"], sparse

    def _rerank(self, question: str, candidates: list[ArticleNode]) -> list[ArticleNode]:
        if not candidates:
            return []
        reranker = self._reranker_model()
        pairs = [[question, _rerank_text(candidate, max_chars=self.config.rerank_max_chars)] for candidate in candidates]
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

    def _guidance_hits(self, question: str, plan: LegalQueryPlan, *, limit: int) -> list[ArticleNode]:
        if self.lexical_index is None:
            return []
        guidance_queries = []
        for target in plan.multi_hop_targets[:2]:
            guidance_queries.append(f"{question} {target} hướng dẫn")
        hits: list[ArticleNode] = []
        seen: set[str] = set()
        for query in guidance_queries:
            for article in self._lexical_search(query, limit):
                if article.article_key in seen:
                    continue
                seen.add(article.article_key)
                hits.append(article)
        return hits

    def _graph_hits(self, question: str, plan: LegalQueryPlan, seed_hits: list[ArticleNode], *, limit: int) -> list[ArticleNode]:
        if self.lexical_index is None:
            self._last_graph_doc_ids = []
            return []
        related_doc_ids: set[str] = set()
        for article in seed_hits:
            relations = article.metadata.get("document_relations", {})
            related_doc_ids.update(str(item) for item in relations.get("guides_doc_ids", []))
            related_doc_ids.update(str(item) for item in relations.get("amends_doc_ids", []))
            related_doc_ids.update(self.reverse_guides_by_doc_id.get(article.doc_id, []))
        if hasattr(self.lexical_index, "related_doc_ids"):
            related_doc_ids.update(self.lexical_index.related_doc_ids({article.doc_id for article in seed_hits}))
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
        hits = self._lexical_search(query or question, max(limit * 2, limit))
        filtered = []
        seen: set[str] = set()
        for hit in hits:
            if hit.doc_id not in related_doc_ids or hit.article_key in seen:
                continue
            seen.add(hit.article_key)
            filtered.append(hit)
        return filtered

    def _lexical_search(self, query: str, limit: int) -> list[ArticleNode]:
        if self.lexical_index is None:
            return []
        if isinstance(self.lexical_index, BM25Index):
            return retrieve_articles(
                self.lexical_index,
                query,
                top_k=limit,
                bm25_top_k=limit,
                min_score_ratio=0.0,
            )
        return self.lexical_index.search(query, top_k=limit)

    def _exact_hits(self, text: str) -> list[ArticleNode]:
        if hasattr(self.lexical_index, "exact_search"):
            return self.lexical_index.exact_search(text, top_k=max(20, self.config.final_top_k * 4))
        return exact_match(self.articles, text) if self.articles else []


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

    if config.qdrant_path and int(preflight["points"]) > config.max_embedded_points:
        raise HybridRetrievalError(
            "embedded_qdrant_too_large:"
            f"points={preflight['points']}:limit={config.max_embedded_points}:"
            "use_qdrant_server_mode_by_setting_qdrant.path_empty"
        )

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


def _budget_for_plan(plan: LegalQueryPlan, config: HybridRetrievalConfig, requested_top_k: int) -> RetrievalBudget:
    exact_targeted = bool(plan.target_doc_ids or plan.target_article_labels)
    many_components = len(plan.requested_components) >= 3
    hard_multi_part = plan.question_type == "comparison" or many_components
    guidance_heavy = plan.needs_guidance_docs or plan.question_type in {"procedure", "penalty"} or plan.intent in {"tax_land", "penalty"}
    broad_list = plan.question_type in {"list", "mixed"} and not plan.target_doc_ids and not plan.domain_anchors

    if exact_targeted:
        desired = {
            "bm25_top_k": 32,
            "vector_top_k": 24,
            "fusion_top_k": 28,
            "rerank_top_k": 12,
            "final_top_k": 4,
            "graph_expand_top_k": 12,
            "margin_delta": 0.14,
            "max_articles_after_cutoff": 4,
        }
    elif hard_multi_part:
        desired = {
            "bm25_top_k": 64,
            "vector_top_k": 56,
            "fusion_top_k": 44,
            "rerank_top_k": 28,
            "final_top_k": 8,
            "graph_expand_top_k": 20,
            "margin_delta": 0.24,
            "max_articles_after_cutoff": 8,
        }
    elif guidance_heavy:
        desired = {
            "bm25_top_k": 56,
            "vector_top_k": 40,
            "fusion_top_k": 36,
            "rerank_top_k": 22,
            "final_top_k": 6,
            "graph_expand_top_k": 18,
            "margin_delta": 0.20,
            "max_articles_after_cutoff": 6,
        }
    elif broad_list:
        desired = {
            "bm25_top_k": 60,
            "vector_top_k": 48,
            "fusion_top_k": 36,
            "rerank_top_k": 24,
            "final_top_k": 6,
            "graph_expand_top_k": 16,
            "margin_delta": 0.22,
            "max_articles_after_cutoff": 6,
        }
    else:
        desired = {
            "bm25_top_k": 48,
            "vector_top_k": 36,
            "fusion_top_k": 32,
            "rerank_top_k": 18,
            "final_top_k": 5,
            "graph_expand_top_k": 14,
            "margin_delta": config.margin_delta,
            "max_articles_after_cutoff": 5,
        }

    final_top_k = max(1, min(requested_top_k, config.final_top_k, int(desired["final_top_k"])))
    return RetrievalBudget(
        bm25_top_k=_positive_min(int(desired["bm25_top_k"]), config.bm25_top_k),
        vector_top_k=_positive_min(int(desired["vector_top_k"]), config.vector_top_k),
        fusion_top_k=max(final_top_k, _positive_min(int(desired["fusion_top_k"]), config.fusion_top_k)),
        rerank_top_k=max(final_top_k, _positive_min(int(desired["rerank_top_k"]), config.rerank_top_k)),
        final_top_k=final_top_k,
        graph_expand_top_k=_positive_min(int(desired["graph_expand_top_k"]), config.graph_expand_top_k),
        margin_delta=float(desired["margin_delta"]),
        min_articles_after_cutoff=max(1, min(config.min_articles_after_cutoff, final_top_k)),
        max_articles_after_cutoff=max(final_top_k, min(config.max_articles_after_cutoff, int(desired["max_articles_after_cutoff"]))),
    )


def _positive_min(desired: int, configured: int) -> int:
    if configured <= 0:
        return 0
    return min(desired, configured)


def _per_query_limit(total_budget: int, query_count: int, *, floor: int) -> int:
    if total_budget <= 0:
        return 0
    return max(min(total_budget, floor), math.ceil(total_budget / max(1, query_count)))


def _dedupe_planned_queries(queries: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for query in queries:
        text = str(getattr(query, "text", "")).strip()
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        output.append(query)
    return output


def _semantic_query_count(queries: list[Any]) -> int:
    return sum(1 for query in queries if str(getattr(query, "kind", "")) in {"core", "component", "guidance", "expanded", "legal_terms", "regime", "semantic"})


def _query_uses_qdrant(query: Any, semantic_query_count: int) -> bool:
    kind = str(getattr(query, "kind", ""))
    if kind in {"core", "component", "guidance", "expanded", "legal_terms", "regime", "semantic"}:
        return True
    return kind == "original" and semantic_query_count == 0


def _rerank_query_for_plan(question: str, plan: LegalQueryPlan) -> str:
    phrases = [
        plan.normalized_question or question,
        *plan.legal_terms[:4],
        *plan.legal_facets[:4],
        *plan.requested_components[:4],
        *plan.governing_doc_hints[:2],
        *plan.target_doc_ids[:2],
        *plan.target_article_labels[:2],
    ]
    output: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        text = " ".join(str(phrase).split()).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    words = " ".join(output).split()
    return " ".join(words[:48])


def _targeted_component_query(question: str, plan: LegalQueryPlan, component: str) -> str:
    phrases = [
        *plan.must_keep_phrases[:3],
        *plan.entities.get("subjects", [])[:2],
        *plan.entities.get("actions", [])[:2],
        component,
        *plan.target_doc_ids[:1],
        *plan.target_article_labels[:1],
        *plan.lexical_expansions[:2],
    ]
    output: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        value = " ".join(str(phrase).split()).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    if not output:
        output = [question, component]
    return " ".join(" ".join(output).split()[:30])


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
    by_key: dict[str, ArticleNode] = {}
    order: list[str] = []
    for article in articles:
        promoted = _promote_article(article)
        existing = by_key.get(promoted.article_key)
        if existing is None:
            by_key[promoted.article_key] = promoted
            order.append(promoted.article_key)
            continue
        by_key[promoted.article_key] = _merge_duplicate_article_hit(existing, promoted)
    return [by_key[key] for key in order]


def _merge_duplicate_article_hit(left: ArticleNode, right: ArticleNode) -> ArticleNode:
    richer, other = (left, right)
    if _article_richness(right) > _article_richness(left):
        richer, other = right, left
    merged = ArticleNode.from_dict(richer.to_dict())
    merged.score = max(float(left.score), float(right.score))
    metadata = dict(merged.metadata)
    other_trace = str(other.metadata.get("retrieval_trace") or "")
    if other_trace:
        existing_trace = str(metadata.get("retrieval_trace") or "")
        metadata["retrieval_trace"] = f"{existing_trace} | {other_trace}" if existing_trace else other_trace
    if not metadata.get("support_snippet"):
        metadata["support_snippet"] = other.metadata.get("support_snippet", "")
    if not metadata.get("support_span_key"):
        metadata["support_span_key"] = other.metadata.get("support_span_key", "")
    if not metadata.get("support_span_label"):
        metadata["support_span_label"] = other.metadata.get("support_span_label", "")
    merged.metadata = metadata
    return merged


def _article_richness(article: ArticleNode) -> int:
    richness = min(len(article.text), 4000)
    if article.metadata.get("clause_nodes"):
        richness += 5000
    node_type = str(article.metadata.get("node_type") or article.metadata.get("chunk_type") or "article").lower()
    if node_type == "article":
        richness += 2000
    elif node_type == "clause":
        richness += 700
    return richness


def _apply_retrieval_bias(articles: list[ArticleNode], plan: LegalQueryPlan) -> list[ArticleNode]:
    adjusted: list[ArticleNode] = []
    for article in articles:
        clone = ArticleNode.from_dict(article.to_dict())
        clone = _attach_plan_support_span(clone, plan)
        clone.score = clone.score + _bias_delta(clone, plan)
        adjusted.append(clone)
    adjusted.sort(key=lambda item: item.score, reverse=True)
    return adjusted


def _bias_delta(article: ArticleNode, plan: LegalQueryPlan) -> float:
    support = str(article.metadata.get("support_snippet") or "")
    title = f"{article.title_for_submission} {article.article_title} {support[:700]} {article.text[:400]}".lower()
    doc_title = article.title_for_submission.lower()
    question_scope = " ".join(
        [
            plan.normalized_question,
            " ".join(plan.legal_terms),
            " ".join(plan.legal_facets),
            " ".join(plan.target_doc_aliases),
        ]
    ).lower()
    delta = 0.0
    norm_roles = {str(role).lower() for role in article.metadata.get("norm_roles", [])}
    node_type = str(article.metadata.get("node_type") or article.metadata.get("chunk_type") or "article").lower()
    must_terms = [term.lower() for term in plan.filters.get("must_include_terms", []) if term]
    should_terms = [term.lower() for term in plan.filters.get("should_include_terms", []) if term]
    matched_must = sum(1 for term in must_terms if _term_matches_evidence(term, title))
    matched_should = sum(1 for term in should_terms if term in title)
    governing_hints = [hint.lower() for hint in plan.governing_doc_hints if hint]
    matched_governing = sum(1 for hint in governing_hints if hint in title)
    provenance_terms = _specific_provenance_terms(plan)
    matched_provenance = sum(1 for term in provenance_terms if term in title)

    if not _question_mentions_local_scope(question_scope):
        if _is_local_or_pilot_document(article):
            delta -= 0.9
        if article.doc_type.lower() == "nghị quyết" and any(term in doc_title for term in ("hội đồng nhân dân", "hđnd", "địa bàn")):
            delta -= 0.7

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
    if provenance_terms:
        if matched_provenance:
            delta += min(0.6, 0.18 * matched_provenance)
        else:
            delta -= min(0.45, 0.12 * len(provenance_terms))
    delta += _component_coverage_delta(article, plan)
    delta += _support_span_delta(article, plan)
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
        if _has_direct_authority_signal(title):
            delta += 0.35
        elif _is_cross_reference_only_authority_text(title):
            delta -= 0.75
    if plan.needs_guidance_docs:
        if article.doc_type.lower() in {"nghị định", "thông tư"}:
            delta += 0.15
        elif article.doc_type.lower() == "luật":
            delta -= 0.05
    return delta


def _attach_plan_support_span(article: ArticleNode, plan: LegalQueryPlan) -> ArticleNode:
    if article.metadata.get("support_snippet") and article.metadata.get("support_span_key") and not _plan_prefers_parent_list_span(plan):
        requested = {component.lower() for component in plan.requested_components}
        covered = _span_component_coverage(str(article.metadata.get("support_snippet") or ""), plan)
        if not requested or requested.issubset(covered):
            return article
    spans = _best_support_spans(article, plan)
    if not spans:
        return article
    clone = ArticleNode.from_dict(article.to_dict())
    metadata = dict(clone.metadata)
    metadata["support_snippet"] = "\n...\n".join(str(span["text"]).strip() for span in spans if str(span["text"]).strip())
    metadata["support_span_key"] = " + ".join(str(span["span_key"]) for span in spans if str(span["span_key"]).strip())
    metadata["support_span_label"] = " + ".join(str(span["label"]) for span in spans if str(span["label"]).strip())
    metadata["support_span_score"] = round(max(float(span["score"]) for span in spans), 3)
    metadata["support_span_components"] = sorted({component for span in spans for component in span.get("components", [])})
    clone.metadata = metadata
    return clone


def _best_support_spans(article: ArticleNode, plan: LegalQueryPlan) -> list[dict[str, Any]]:
    spans = _iter_article_spans(article)
    if not spans:
        return []
    scored: list[dict[str, Any]] = []
    for span in spans:
        score = _span_relevance(str(span["text"]), plan)
        if score >= 1.2:
            selection_score = score - min(len(str(span["text"])) / 1200.0, 1.2)
            scored.append(
                {
                    **span,
                    "score": score,
                    "selection_score": selection_score,
                    "components": _span_component_coverage(str(span["text"]), plan),
                }
            )
    if not scored:
        return []
    scored.sort(key=lambda item: (-float(item["selection_score"]), -float(item["score"]), len(str(item["text"]))))
    selected = [scored[0]]
    covered = set(scored[0].get("components", []))
    requested = {component.lower() for component in plan.requested_components}
    if len(requested) > 1:
        for span in scored[1:]:
            components = set(span.get("components", []))
            if components - covered:
                selected.append(span)
                covered.update(components)
            if len(selected) >= 2 or requested.issubset(covered):
                break
    return selected


def _iter_article_spans(article: ArticleNode) -> list[dict[str, str]]:
    spans: list[dict[str, str]] = []
    for clause in article.metadata.get("clause_nodes", []) or []:
        clause_text = str(clause.get("text") or "").strip()
        clause_key = str(clause.get("span_key") or "")
        clause_label = str(clause.get("label") or "")
        if clause_text:
            spans.append({"text": clause_text, "span_key": clause_key, "label": clause_label})
        for point in clause.get("point_nodes", []) or []:
            point_text = str(point.get("text") or "").strip()
            if not point_text:
                continue
            parent_prefix = clause_text.splitlines()[0][:260] if clause_text else ""
            text = f"{parent_prefix}\n{point_text}".strip() if parent_prefix and parent_prefix not in point_text else point_text
            spans.append(
                {
                    "text": text,
                    "span_key": str(point.get("span_key") or clause_key),
                    "label": str(point.get("label") or clause_label),
                }
            )
    if spans:
        return spans
    return _fallback_clause_spans_from_text(article)


def _plan_prefers_parent_list_span(plan: LegalQueryPlan) -> bool:
    question_text = " ".join(
        [
            plan.normalized_question,
            " ".join(plan.legal_terms),
            " ".join(plan.legal_facets),
            " ".join(plan.requested_components),
        ]
    ).lower()
    return "hồ sơ" in question_text and any(term in question_text for term in ("gồm", "bao gồm", "những gì"))


def _fallback_clause_spans_from_text(article: ArticleNode) -> list[dict[str, str]]:
    text = str(article.text or "").strip()
    if not text:
        return []
    spans: list[dict[str, str]] = []
    pattern = re.compile(r"(?:^|\n)(\d+)\.\s+(.+?)(?=(?:\n\d+\.\s+)|\Z)", re.DOTALL)
    for number, body in pattern.findall(text):
        span_text = f"{number}. {body}".strip()
        if len(span_text) < 20:
            continue
        spans.append(
            {
                "text": span_text,
                "span_key": f"{article.article_label}|Khoản {number}",
                "label": f"Khoản {number}",
            }
        )
    return spans


def _span_relevance(text: str, plan: LegalQueryPlan) -> float:
    lowered = text.lower()
    score = 0.0
    requested = {component.lower() for component in plan.requested_components}
    for component in requested:
        if component_matches(component, lowered, evidence=True):
            definition = COMPONENT_REGISTRY.get(component)
            score += float(definition.retrieval_weight if definition else 1.0)
    if "mức phạt" in requested:
        if "phạt tiền" in lowered or "mức phạt" in lowered:
            score += 0.8
        if re.search(r"từ\s+\d[\d\.\s]*\s*đồng", lowered):
            score += 1.2
    if "hồ sơ" in requested and component_matches("hồ sơ", lowered, evidence=True):
        question_text = " ".join(
            [
                plan.normalized_question,
                " ".join(plan.legal_terms),
                " ".join(plan.legal_facets),
                " ".join(plan.requested_components),
            ]
        ).lower()
        proposal_like = any(term in question_text for term in ("đề nghị", "đề xuất", "nhu cầu hỗ trợ"))
        if proposal_like and any(term in lowered for term in ("hồ sơ đề xuất nhu cầu hỗ trợ", "đề xuất nhu cầu hỗ trợ")):
            score += 2.1
        if proposal_like and "hồ sơ thanh toán" in lowered:
            score -= 1.4
        if "cụm liên kết ngành" in question_text and "cụm liên kết ngành" in lowered:
            score += 0.6
        if any(term in question_text for term in ("gồm", "bao gồm", "những gì")) and _contains_lettered_list(lowered):
            score += 2.0
    weighted_phrases: list[tuple[str, float]] = []
    for item in plan.entities.get("actions", []):
        weighted_phrases.append((item, 1.0))
    for item in plan.entities.get("objects", []):
        weighted_phrases.append((item, 0.9))
    for item in plan.entities.get("subjects", []):
        weighted_phrases.append((item, 0.45))
    for item in [*plan.legal_terms[:5], *plan.legal_facets[:5]]:
        weighted_phrases.append((item, 0.35))

    seen: set[str] = set()
    for phrase, weight in weighted_phrases:
        normalized = " ".join(normalize_legal_query_text(str(phrase)).lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in lowered:
            score += weight
            continue
        score += weight * _legal_phrase_overlap(normalized, lowered)

    return score


def _legal_phrase_overlap(phrase: str, text: str) -> float:
    stopwords = {
        "của",
        "cho",
        "và",
        "hoặc",
        "khi",
        "theo",
        "được",
        "bị",
        "phải",
        "có",
        "một",
        "những",
        "các",
    }
    phrase_tokens = {
        token for token in re.findall(r"[\wÀ-ỹ]+", phrase) if len(token) > 1 and token not in stopwords
    }
    if len(phrase_tokens) < 2:
        return 0.0
    normalized_text = normalize_legal_query_text(text).lower()
    text_tokens = set(re.findall(r"[\wÀ-ỹ]+", normalized_text))
    overlap = len(phrase_tokens & text_tokens)
    coverage = overlap / len(phrase_tokens)
    if overlap < 2 or coverage < 0.45:
        return 0.0
    return min(0.8, coverage * 0.8)


def _specific_provenance_terms(plan: LegalQueryPlan) -> list[str]:
    values = [
        *plan.governing_doc_hints[:4],
        *plan.domain_anchors[:3],
        *plan.must_keep_phrases[:4],
    ]
    output: list[str] = []
    for value in values:
        normalized = " ".join(str(value).lower().replace("_", " ").split())
        if len(normalized) < 5 or normalized in {"hỗ trợ", "điều kiện", "trách nhiệm", "xử phạt"}:
            continue
        if normalized not in output:
            output.append(normalized)
    return output


def _span_component_coverage(text: str, plan: LegalQueryPlan) -> set[str]:
    requested = {component.lower() for component in plan.requested_components}
    return {component for component in requested if component_matches(component, text, evidence=True)}


def _contains_lettered_list(text: str) -> bool:
    return len(re.findall(r"(?:^|\n)\s*[a-zđ]\)\s+", text, flags=re.IGNORECASE)) >= 2


def _is_cross_reference_only_authority_text(text: str) -> bool:
    lowered = text.lower()
    pointer_markers = (
        "được thực hiện theo quy định của pháp luật",
        "thực hiện theo quy định của pháp luật",
        "áp dụng quy định của pháp luật",
        "theo quy định của pháp luật về xử phạt",
    )
    if not any(marker in lowered for marker in pointer_markers):
        return False
    return "thẩm quyền" in lowered or "xử phạt" in lowered


def _has_direct_authority_signal(text: str) -> bool:
    lowered = text.lower()
    direct_patterns = (
        "có thẩm quyền",
        "thẩm quyền của",
        "chủ tịch ủy ban",
        "chủ tịch ubnd",
        "chánh thanh tra",
        "thanh tra viên",
        "cục trưởng",
        "chi cục trưởng",
        "tổng cục trưởng",
        "trưởng đoàn thanh tra",
        "cơ quan thuế có thẩm quyền",
        "cơ quan quản lý thuế có thẩm quyền",
    )
    return any(pattern in lowered for pattern in direct_patterns)


def _support_span_delta(article: ArticleNode, plan: LegalQueryPlan) -> float:
    score = float(article.metadata.get("support_span_score") or 0.0)
    if score <= 0:
        return 0.0
    delta = min(0.55, score * 0.08)
    if "mức phạt" in {component.lower() for component in plan.requested_components}:
        snippet = str(article.metadata.get("support_snippet") or "").lower()
        if "phạt tiền" in snippet and re.search(r"từ\s+\d[\d\.\s]*\s*đồng", snippet):
            delta += 0.35
    return delta


def _component_coverage_delta(article: ArticleNode, plan: LegalQueryPlan) -> float:
    if not plan.requested_components:
        return 0.0
    support = str(article.metadata.get("support_snippet") or "")
    text = f"{article.title_for_submission} {article.article_title} {support[:900]} {article.text[:1400]}".lower()
    delta = 0.0
    for component in plan.requested_components:
        definition = COMPONENT_REGISTRY.get(component.lower())
        if definition is None:
            continue
        if component_matches(component, text, evidence=True):
            delta += 0.2 * definition.retrieval_weight
        else:
            delta -= 0.32 * definition.retrieval_weight
    return delta


def _term_matches_evidence(term: str, text: str) -> bool:
    if term in COMPONENT_REGISTRY:
        return component_matches(term, text, evidence=True)
    return term in text


def _question_mentions_local_scope(text: str) -> bool:
    return any(
        term in text
        for term in (
            "địa bàn",
            "tỉnh",
            "thành phố",
            "hội đồng nhân dân",
            "hđnd",
            "ủy ban nhân dân",
            "ubnd",
            "thủ đô",
        )
    )


def _is_local_or_pilot_document(article: ArticleNode) -> bool:
    title = article.title_for_submission.lower()
    if any(term in title for term in ("hội đồng nhân dân", "hđnd", "ủy ban nhân dân", "ubnd", "địa bàn tỉnh", "địa bàn thành phố")):
        return True
    if any(term in title for term in ("thí điểm", "đặc thù phát triển", "việt nam - hàn quốc", "thành phố cần thơ", "tỉnh yên bái")):
        return True
    return False


def _suppress_unrequested_local_scope(
    articles: list[ArticleNode], plan: LegalQueryPlan, *, question: str | None = None
) -> tuple[list[ArticleNode], set[str]]:
    if len(articles) < 2:
        return articles, set()
    question_scope = (question or plan.normalized_question).lower()
    if _question_mentions_local_scope(question_scope):
        return articles, set()
    national = [article for article in articles if not _is_local_or_pilot_document(article)]
    local = [article for article in articles if _is_local_or_pilot_document(article)]
    if not national or not local:
        return articles, set()
    required = {component.lower() for component in plan.requested_components}
    national_coverage: set[str] = set()
    for article in national:
        national_coverage.update(
            _span_component_coverage(
                f"{article.article_title} {article.metadata.get('support_snippet', '')} {article.text[:1600]}",
                plan,
            )
        )
    if required and not required.issubset(national_coverage):
        return articles, set()
    removed = {article.doc_id for article in local}
    return national, removed


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
        rerank_max_chars=int(reranker.get("max_chars", retrieval.get("rerank_max_chars", 2800))),
        max_payload_text_chars=int(retrieval.get("max_payload_text_chars", 20000)),
        upsert_batch_size=int(retrieval.get("upsert_batch_size", 512)),
        load_bm25=bool(retrieval.get("load_bm25", True)),
        lexical_index_path=str(retrieval.get("lexical_index", "") or ""),
        max_embedded_points=int(qdrant.get("max_embedded_points", retrieval.get("max_embedded_points", 20_000))),
    )


def _new_qdrant_client(url: str, path: str = "") -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        if path:
            raise HybridRetrievalError("missing_dependency:qdrant-client; embedded qdrant path requires optional dependency 'rag'") from exc
        return _QdrantHttpClient(url)
    if path:
        return QdrantClient(path=path)
    return QdrantClient(url=url)


class _QdrantHttpClient:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def collection_exists(self, collection: str) -> bool:
        try:
            self._request("GET", f"/collections/{collection}")
            return True
        except HybridRetrievalError as exc:
            if "qdrant_http_error:404" in str(exc):
                return False
            raise

    def delete_collection(self, collection: str) -> None:
        self._request("DELETE", f"/collections/{collection}")

    def create_collection(self, collection: str, vectors_config: Any, sparse_vectors_config: Any) -> None:
        self._request(
            "PUT",
            f"/collections/{collection}",
            {
                "vectors": vectors_config,
                "sparse_vectors": sparse_vectors_config,
            },
        )

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        payload_points = []
        for point in points:
            if isinstance(point, dict):
                payload_points.append(point)
            else:
                payload_points.append(
                    {
                        "id": str(point.id),
                        "vector": point.vector,
                        "payload": point.payload,
                    }
                )
        self._request("PUT", f"/collections/{collection_name}/points?wait=true", {"points": payload_points})

    def query_points(self, collection_name: str, query: Any, using: str, with_payload: bool, limit: int) -> Any:
        data = self._request(
            "POST",
            f"/collections/{collection_name}/points/query",
            {
                "query": query,
                "using": using,
                "with_payload": with_payload,
                "limit": limit,
            },
        )
        points = data.get("result", {}).get("points", data.get("result", []))
        return type("QdrantHttpResult", (), {"points": points})()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HybridRetrievalError(f"qdrant_http_error:{exc.code}:{detail[:300]}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise HybridRetrievalError(f"qdrant_http_connection_error:{exc}") from exc
        return json.loads(raw) if raw else {}


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


def _margin_cutoff_with_budget(articles: list[ArticleNode], budget: RetrievalBudget) -> tuple[list[ArticleNode], str]:
    if not articles:
        return [], ""
    kept = [articles[0]]
    cutoff_reason = ""
    for previous, current in zip(articles, articles[1:], strict=False):
        if len(kept) >= budget.max_articles_after_cutoff:
            cutoff_reason = f"max_articles:{budget.max_articles_after_cutoff}"
            break
        margin = previous.score - current.score
        if len(kept) >= budget.min_articles_after_cutoff and margin > budget.margin_delta:
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


def _rerank_text(article: ArticleNode, max_chars: int) -> str:
    support = str(article.metadata.get("support_snippet") or "").strip()
    body = support or article.text
    return _truncate_text(
        "\n".join([article.relevant_article, article.article_title, body]),
        max_chars,
    )


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
    return Path(config.embedding_cache_dir) / f"{digest.hexdigest()[:24]}.jsonl.gz"


def _cache_path_from_digest(digest: str | int, config: HybridRetrievalConfig) -> Path:
    return Path(config.embedding_cache_dir) / f"{digest}.jsonl.gz"


def _is_gzip_cache(path: Path) -> bool:
    return ".gz" in path.suffixes


def _open_cache_text(path: Path, mode: str):
    if _is_gzip_cache(path):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _embedding_cache_valid_prefix(
    path: Path,
    articles: list[ArticleNode],
    config: HybridRetrievalConfig,
) -> int:
    if not path.exists():
        return 0
    count = 0
    with _open_cache_text(path, "rt") as f:
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
    with _open_cache_text(path, "rt") as f:
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
    with _open_cache_text(path, "rt") as f:
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
    with _open_cache_text(path, "rt") as f:
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
    tmp_path = path.with_name(path.name + ".tmp")
    with _open_cache_text(path, "rt") as src, _open_cache_text(tmp_path, "wt") as dst:
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
    with _open_cache_text(path, "at") as f:
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
    with _open_cache_text(path, "wt") as f:
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
