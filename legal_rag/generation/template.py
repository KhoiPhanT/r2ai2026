from __future__ import annotations

from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import ARTICLE_PATTERN, compact_snippet


def generate_grounded_answer(question: str, articles: list[ArticleNode], max_articles: int = 5) -> str:
    if not articles:
        return (
            "Không tìm thấy căn cứ pháp luật đủ chắc trong corpus đã lập chỉ mục. "
            "Cần bổ sung hoặc kiểm tra lại văn bản pháp luật chính thống trước khi trả lời."
        )

    chosen = articles[:max_articles]
    lead = "Theo các căn cứ đã truy hồi, câu trả lời sơ bộ là: "
    evidence_sentences = []
    for article in chosen[:3]:
        snippet = _mask_unselected_article_refs(
            compact_snippet(article.text, max_chars=260),
            allowed_labels={item.article_label for item in chosen},
        )
        evidence_sentences.append(f"{article.article_label} {article.title_for_submission} quy định: {snippet}")
    citations = "; ".join(f"{a.article_label} {a.title_for_submission}" for a in chosen)
    return lead + " ".join(evidence_sentences) + f"\nCăn cứ: {citations}."


def _mask_unselected_article_refs(text: str, allowed_labels: set[str]) -> str:
    allowed = {label.lower() for label in allowed_labels}

    def replace(match) -> str:
        label = f"Điều {match.group(1)}"
        if label.lower() in allowed:
            return label
        return "điều liên quan"

    return ARTICLE_PATTERN.sub(replace, text)
