from __future__ import annotations

from legal_rag.retrieval.bm25 import BM25Index
from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import extract_article_labels, extract_doc_ids, normalize_for_match


def retrieve_articles(
    index: BM25Index,
    question: str,
    top_k: int = 8,
    bm25_top_k: int = 40,
    min_score_ratio: float = 0.08,
) -> list[ArticleNode]:
    scored: dict[str, ArticleNode] = {}

    for hit in index.search(question, top_k=bm25_top_k):
        scored[hit.article_key] = hit

    for hit in exact_match(index.articles, question):
        current = scored.get(hit.article_key)
        if current is None or hit.score > current.score:
            scored[hit.article_key] = hit

    results = sorted(scored.values(), key=lambda article: article.score, reverse=True)
    if results and min_score_ratio > 0:
        threshold = results[0].score * min_score_ratio
        results = [article for article in results if article.score >= threshold]
    return results[:top_k]


def exact_match(articles: list[ArticleNode], question: str) -> list[ArticleNode]:
    q_norm = normalize_for_match(question)
    doc_ids = {doc_id.upper() for doc_id in extract_doc_ids(question)}
    article_labels = {label.lower() for label in extract_article_labels(question)}
    hits: list[ArticleNode] = []

    for article in articles:
        score = 0.0
        title_norm = normalize_for_match(article.title_for_submission)
        doc_matched = article.doc_id.upper() in doc_ids
        title_matched = title_norm and title_norm in q_norm
        type_matched = bool(article.doc_type and article.doc_type.lower() in q_norm and doc_matched)
        if doc_matched:
            score += 200.0
        if article.article_label.lower() in article_labels and (doc_matched or title_matched):
            score += 120.0
        elif article.article_label.lower() in article_labels:
            score += 8.0
        if title_matched:
            score += 80.0
        elif type_matched:
            score += 50.0
        if score:
            clone = ArticleNode.from_dict(article.to_dict())
            clone.score = score
            hits.append(clone)
    return hits
