from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import extract_article_labels


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

    return VerificationResult(ok=not issues, issues=issues)


def _answer_disambiguates(answer: str, articles: list[ArticleNode]) -> bool:
    lowered = answer.lower()
    return all(article.doc_id.lower() in lowered or article.title_for_submission.lower() in lowered for article in articles)
