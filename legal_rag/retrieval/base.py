from __future__ import annotations

from typing import Protocol

from legal_rag.schemas.models import ArticleNode


class SearchBackend(Protocol):
    backend_name: str
    last_trace: dict

    def search(self, query: str, top_k: int = 20) -> list[ArticleNode]: ...

    def get_article(self, article_key: str) -> ArticleNode | None: ...

    def articles_for_doc_ids(self, doc_ids: set[str], limit: int = 100) -> list[ArticleNode]: ...

