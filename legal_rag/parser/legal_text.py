from __future__ import annotations

from legal_rag.schemas.models import ArticleNode, DocumentRecord
from legal_rag.utils.text import ARTICLE_HEADING_PATTERN, normalize_text


def parse_document(doc: DocumentRecord) -> list[ArticleNode]:
    text = normalize_text(doc.raw_text)
    headings = list(ARTICLE_HEADING_PATTERN.finditer(text))
    if not headings:
        return [_article_from_span(doc, "Điều 1", "", text, synthetic=True)]

    articles: list[ArticleNode] = []
    for index, match in enumerate(headings):
        start = match.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        article_label = _canonical_article_label(match.group(1))
        article_title = normalize_text(match.group(2))
        article_text = normalize_text(text[start:end])
        articles.append(_article_from_span(doc, article_label, article_title, article_text))
    return articles


def _canonical_article_label(raw: str) -> str:
    suffix = raw.split(None, 1)[1].strip()
    return f"Điều {suffix}"


def _article_from_span(
    doc: DocumentRecord,
    article_label: str,
    article_title: str,
    text: str,
    synthetic: bool = False,
) -> ArticleNode:
    article_key = f"{doc.doc_id}|{doc.title_for_submission}|{article_label}"
    metadata = dict(doc.metadata)
    metadata["synthetic_article"] = synthetic
    return ArticleNode(
        article_key=article_key,
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        title_for_submission=doc.title_for_submission,
        article_label=article_label,
        article_title=article_title,
        text=text,
        status=doc.status,
        source_url=doc.source_url,
        metadata=metadata,
    )

