from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from legal_rag.question_metadata import infer_legal_facets, infer_must_include_terms
from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import extract_article_labels

FACET_STOPWORDS = {"và", "các", "những", "theo", "của", "cho", "khi", "với"}


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    issues: list[str]


def verify_prediction_evidence(answer: str, articles: list[ArticleNode]) -> VerificationResult:
    issues: list[str] = []
    cited = {label.lower() for label in extract_article_labels(answer)}
    available = {article.article_label.lower() for article in articles}

    if articles and not cited:
        issues.append("answer_missing_article_citation")
    for label in sorted(cited):
        if label not in available:
            issues.append(f"citation_not_in_retrieved_set:{label}")

    article_keys = [article.article_key for article in articles]
    if len(article_keys) != len(set(article_keys)):
        issues.append("duplicate_article_key")

    return VerificationResult(ok=not issues, issues=issues)


def verify_used_evidence_answer(
    answer: str,
    used_articles: list[ArticleNode],
    candidate_articles: Iterable[ArticleNode] | None = None,
    question: str = "",
) -> VerificationResult:
    issues: list[str] = []
    if not used_articles:
        issues.append("answer_missing_used_evidence")
        return VerificationResult(ok=False, issues=issues)

    cited = [label.lower() for label in extract_article_labels(answer)]
    if not cited:
        issues.append("answer_missing_article_citation")
    used_by_label: dict[str, list[ArticleNode]] = {}
    for article in used_articles:
        used_by_label.setdefault(article.article_label.lower(), []).append(article)

    candidate_keys = {article.article_key for article in candidate_articles or []}
    for article in used_articles:
        if candidate_keys and article.article_key not in candidate_keys:
            issues.append(f"used_evidence_not_retrieved:{article.article_key}")

    for label in sorted(set(cited)):
        matches = used_by_label.get(label, [])
        if not matches:
            issues.append(f"citation_not_in_used_evidence:{label}")
        elif len(matches) > 1 and not _answer_disambiguates(answer, matches):
            issues.append(f"ambiguous_article_label:{label}")
    cited_set = set(cited)
    for article in used_articles:
        if article.article_label.lower() not in cited_set:
            issues.append(f"used_evidence_not_cited:{article.article_key}")

    article_keys = [article.article_key for article in used_articles]
    if len(article_keys) != len(set(article_keys)):
        issues.append("duplicate_article_key")

    semantic_issues = _semantic_sanity_issues(question, used_articles)
    issues.extend(semantic_issues)

    return VerificationResult(ok=not issues, issues=issues)


def _answer_disambiguates(answer: str, articles: list[ArticleNode]) -> bool:
    lowered = answer.lower()
    return all(article.doc_id.lower() in lowered or article.title_for_submission.lower() in lowered for article in articles)


def _semantic_sanity_issues(question: str, used_articles: list[ArticleNode]) -> list[str]:
    if not question or not used_articles:
        return []
    combined = " ".join(
        f"{article.title_for_submission} {article.article_title} {article.text[:1200]}".lower()
        for article in used_articles
    )
    anchors = infer_must_include_terms(question)
    if anchors:
        matched = [anchor for anchor in anchors if anchor.lower() in combined]
        if not matched:
            return [f"semantic_domain_mismatch:{'|'.join(anchors[:3])}"]
    facets = [facet for facet in infer_legal_facets(question) if len(facet) > 3]
    if facets:
        matched_facets = [
            facet
            for facet in facets
            if facet.lower() in combined
            or any(token in combined for token in facet.lower().split() if len(token) > 1 and token not in FACET_STOPWORDS)
        ]
        if not matched_facets:
            return [f"semantic_facet_mismatch:{'|'.join(facets[:3])}"]
    return []
