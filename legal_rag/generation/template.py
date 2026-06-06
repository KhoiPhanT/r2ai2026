from __future__ import annotations

from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import compact_snippet


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
        snippet = compact_snippet(article.text, max_chars=260)
        evidence_sentences.append(f"{article.article_label} {article.title_for_submission} quy định: {snippet}")
    citations = "; ".join(f"{a.article_label} {a.title_for_submission}" for a in chosen)
    return lead + " ".join(evidence_sentences) + f"\nCăn cứ: {citations}."

