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
class MetadataSignal:
    name: str
    value: str
    source: str
    source_text: str = ""
    confidence: float = 0.0
    enforcement: str = "advisory"
    evidence_validated: bool = False
    repair_round: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadataSignal":
        return cls(
            name=str(data.get("name") or ""),
            value=str(data.get("value") or ""),
            source=str(data.get("source") or ""),
            source_text=str(data.get("source_text") or ""),
            confidence=max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
            enforcement=str(data.get("enforcement") or "advisory"),
            evidence_validated=bool(data.get("evidence_validated")),
            repair_round=int(data.get("repair_round") or 0),
        )


@dataclass(slots=True)
class RequestedComponent:
    name: str
    requirement: str
    source_span: str = ""
    confidence: float = 0.0
    source: str = "question_rule"
    evidence_validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequestedComponent":
        requirement = str(data.get("requirement") or "optional")
        if requirement not in {"required", "optional", "not_requested"}:
            requirement = "optional"
        return cls(
            name=str(data.get("name") or ""),
            requirement=requirement,
            source_span=str(data.get("source_span") or ""),
            confidence=max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
            source=str(data.get("source") or "question_rule"),
            evidence_validated=bool(data.get("evidence_validated")),
        )


@dataclass(slots=True)
class EvidenceMetadata:
    article_key: str
    supported_components: list[str] = field(default_factory=list)
    norm_roles: list[str] = field(default_factory=list)
    support_type: str = "direct"
    subjects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    time_or_amount: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    lexical_expansions: list[str] = field(default_factory=list)
    candidate_regimes: list[str] = field(default_factory=list)
    must_keep_phrases: list[str] = field(default_factory=list)
    planned_queries: list[str] = field(default_factory=list)
    signals: list[MetadataSignal] = field(default_factory=list)
    component_requirements: list[RequestedComponent] = field(default_factory=list)
    reconciliation_issues: list[str] = field(default_factory=list)
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
            lexical_expansions=[str(item) for item in data.get("lexical_expansions", []) if str(item).strip()],
            candidate_regimes=[str(item) for item in data.get("candidate_regimes", []) if str(item).strip()],
            must_keep_phrases=[str(item) for item in data.get("must_keep_phrases", []) if str(item).strip()],
            planned_queries=[str(item) for item in data.get("planned_queries", []) if str(item).strip()],
            signals=[MetadataSignal.from_dict(item) for item in data.get("signals", []) if isinstance(item, dict)],
            component_requirements=[
                RequestedComponent.from_dict(item) for item in data.get("component_requirements", []) if isinstance(item, dict)
            ],
            reconciliation_issues=[str(item) for item in data.get("reconciliation_issues", []) if str(item).strip()],
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
    planner_reasoning: bool = False
    answer_reasoning: bool = False
    planner_num_ctx: int = 0
    answer_num_ctx: int = 0
    actual_backend: str = ""
    timeout_stage: str = ""
    repair_attempts: list[dict[str, Any]] = field(default_factory=list)
    evidence_metadata: list[EvidenceMetadata] = field(default_factory=list)
    verification_warnings: list[str] = field(default_factory=list)

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
