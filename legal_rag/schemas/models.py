from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Question:
    id: int
    question: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
class GoldQuestionMetadata:
    question_id: int
    intent: str = ""
    question_type: str = ""
    answer_shape: str = ""
    needs_guidance_docs: bool = False
    gold_relevant_docs: list[str] = field(default_factory=list)
    gold_relevant_articles: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldQuestionMetadata":
        return cls(
            question_id=int(data.get("question_id") or data.get("id") or 0),
            intent=str(data.get("intent") or ""),
            question_type=str(data.get("question_type") or ""),
            answer_shape=str(data.get("answer_shape") or ""),
            needs_guidance_docs=bool(data.get("needs_guidance_docs")),
            gold_relevant_docs=[str(item) for item in data.get("gold_relevant_docs", []) if str(item).strip()],
            gold_relevant_articles=[str(item) for item in data.get("gold_relevant_articles", []) if str(item).strip()],
            notes=str(data.get("notes") or ""),
        )


@dataclass(slots=True)
class PredictedQuestionMetadata:
    intent: str
    question_type: str
    answer_shape: str
    needs_guidance_docs: bool
    target_doc_ids: list[str] = field(default_factory=list)
    target_article_labels: list[str] = field(default_factory=list)
    legal_facets: list[str] = field(default_factory=list)
    retrieval_bias: str = ""
    requested_components: list[str] = field(default_factory=list)
    target_norm_roles: list[str] = field(default_factory=list)
    governing_doc_hints: list[str] = field(default_factory=list)
    domain_anchors: list[str] = field(default_factory=list)
    legal_subjects: list[str] = field(default_factory=list)
    legal_actions: list[str] = field(default_factory=list)
    legal_objects: list[str] = field(default_factory=list)
    time_or_amount: list[str] = field(default_factory=list)
    planned_queries: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictedQuestionMetadata":
        return cls(
            intent=str(data.get("intent") or "general"),
            question_type=str(data.get("question_type") or "mixed"),
            answer_shape=str(data.get("answer_shape") or "single_rule"),
            needs_guidance_docs=bool(data.get("needs_guidance_docs")),
            target_doc_ids=[str(item) for item in data.get("target_doc_ids", []) if str(item).strip()],
            target_article_labels=[str(item) for item in data.get("target_article_labels", []) if str(item).strip()],
            legal_facets=[str(item) for item in data.get("legal_facets", []) if str(item).strip()],
            retrieval_bias=str(data.get("retrieval_bias") or ""),
            requested_components=[str(item) for item in data.get("requested_components", []) if str(item).strip()],
            target_norm_roles=[str(item) for item in data.get("target_norm_roles", []) if str(item).strip()],
            governing_doc_hints=[str(item) for item in data.get("governing_doc_hints", []) if str(item).strip()],
            domain_anchors=[str(item) for item in data.get("domain_anchors", []) if str(item).strip()],
            legal_subjects=[str(item) for item in data.get("legal_subjects", []) if str(item).strip()],
            legal_actions=[str(item) for item in data.get("legal_actions", []) if str(item).strip()],
            legal_objects=[str(item) for item in data.get("legal_objects", []) if str(item).strip()],
            time_or_amount=[str(item) for item in data.get("time_or_amount", []) if str(item).strip()],
            planned_queries=[str(item) for item in data.get("planned_queries", []) if str(item).strip()],
            confidence=float(data.get("confidence") or 0.0),
        )


@dataclass(slots=True)
class QuestionRunTrace:
    id: int
    question: str
    predicted_metadata: PredictedQuestionMetadata
    candidate_counts: dict[str, int] = field(default_factory=dict)
    reranked_evidence: list[str] = field(default_factory=list)
    supporting_spans: list[str] = field(default_factory=list)
    retrieval_path: list[str] = field(default_factory=list)
    graph_expansions: list[str] = field(default_factory=list)
    threshold_cutoff_reason: str = ""
    used_evidence_ids: list[str] = field(default_factory=list)
    verifier_issues: list[str] = field(default_factory=list)
    final_relevant_docs: list[str] = field(default_factory=list)
    final_relevant_articles: list[str] = field(default_factory=list)
    planner_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    answer_ms: float = 0.0
    actual_backend: str = ""
    timeout_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["predicted_metadata"] = self.predicted_metadata.to_dict()
        return data


@dataclass(slots=True)
class FinalPrediction:
    id: int
    question: str
    answer: str
    relevant_docs: list[str]
    relevant_articles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
