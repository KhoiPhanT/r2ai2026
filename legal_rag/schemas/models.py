from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Question:
    id: int
    question: str


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    doc_type: str
    trich_yeu: str
    title_for_submission: str
    raw_text: str
    issuer: str = ""
    issue_date: str = ""
    effective_date: str = ""
    status: str = ""
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_doc(self) -> str:
        return f"{self.doc_id}|{self.title_for_submission}"


@dataclass(slots=True)
class ArticleNode:
    article_key: str
    doc_id: str
    doc_type: str
    title_for_submission: str
    article_label: str
    article_title: str
    text: str
    status: str = ""
    source_url: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def relevant_doc(self) -> str:
        return f"{self.doc_id}|{self.title_for_submission}"

    @property
    def relevant_article(self) -> str:
        return f"{self.doc_id}|{self.title_for_submission}|{self.article_label}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArticleNode":
        return cls(**data)


@dataclass(slots=True)
class RetrievalResult:
    query: str
    articles: list[ArticleNode]


@dataclass(slots=True)
class FinalPrediction:
    id: int
    question: str
    answer: str
    relevant_docs: list[str]
    relevant_articles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

