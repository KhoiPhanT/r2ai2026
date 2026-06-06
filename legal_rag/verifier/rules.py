from __future__ import annotations

from dataclasses import dataclass

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

