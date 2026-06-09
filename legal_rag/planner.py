from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from legal_rag.generation.ollama import OllamaConfig, OllamaError, parse_json_response, request_ollama_chat
from legal_rag.question_metadata import (
    infer_doc_type_hints,
    infer_domain_anchors,
    infer_governing_doc_hints,
    infer_must_include_terms,
    infer_needs_guidance,
    infer_requested_components,
    infer_runtime_metadata,
    infer_target_norm_roles,
)
from legal_rag.schemas.models import PredictedQuestionMetadata
from legal_rag.utils.text import extract_article_labels, extract_doc_ids

PLANNER_INTENTS = {
    "definition",
    "condition",
    "procedure",
    "deadline",
    "penalty",
    "support_policy",
    "tax_land",
    "authority",
    "responsibility",
    "comparison",
    "general",
}
QUERY_KINDS = {"original", "legal_terms", "expanded"}
QUESTION_TYPES = {"definition", "list", "condition", "procedure", "penalty", "deadline", "authority", "comparison", "mixed"}
ANSWER_SHAPES = {"single_rule", "list_items", "penalty_and_remedy", "procedure_steps", "conditions_list", "document_pointer"}
RETRIEVAL_BIASES = {"content_articles", "procedure_articles", "sanction_articles", "authority_articles"}
NORM_ROLES = {"procedure", "authority", "penalty", "remedy", "condition", "support_policy"}


@dataclass(slots=True)
class PlannedQuery:
    kind: str
    text: str
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LegalQueryPlan:
    intent: str
    question_scope: str
    normalized_question: str
    question_type: str = "mixed"
    answer_shape: str = "single_rule"
    legal_terms: list[str] = field(default_factory=list)
    legal_facets: list[str] = field(default_factory=list)
    requested_components: list[str] = field(default_factory=list)
    target_norm_roles: list[str] = field(default_factory=list)
    governing_doc_hints: list[str] = field(default_factory=list)
    domain_anchors: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    target_doc_ids: list[str] = field(default_factory=list)
    target_doc_aliases: list[str] = field(default_factory=list)
    target_article_labels: list[str] = field(default_factory=list)
    queries: list[PlannedQuery] = field(default_factory=list)
    filters: dict[str, list[str]] = field(default_factory=dict)
    retrieval_bias: str = "content_articles"
    needs_guidance_docs: bool = False
    multi_hop_targets: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["queries"] = [query.to_dict() for query in self.queries]
        return data

    def predicted_metadata(self) -> PredictedQuestionMetadata:
        return PredictedQuestionMetadata(
            intent=self.intent,
            question_type=self.question_type,
            answer_shape=self.answer_shape,
            needs_guidance_docs=self.needs_guidance_docs,
            target_doc_ids=self.target_doc_ids,
            target_article_labels=self.target_article_labels,
            legal_facets=self.legal_facets,
            retrieval_bias=self.retrieval_bias,
            requested_components=self.requested_components,
            target_norm_roles=self.target_norm_roles,
            governing_doc_hints=self.governing_doc_hints,
            domain_anchors=self.domain_anchors,
            legal_subjects=self.entities.get("subjects", []),
            legal_actions=self.entities.get("actions", []),
            legal_objects=self.entities.get("objects", []),
            time_or_amount=self.entities.get("amounts_or_deadlines", []),
            planned_queries=[query.text for query in self.queries],
            confidence=self.confidence,
        )


def plan_legal_query(question: str, config: OllamaConfig) -> LegalQueryPlan:
    raw = request_ollama_chat(_planner_system_prompt(), _planner_user_prompt(question), config)
    try:
        data = parse_json_response(raw, "planner")
    except OllamaError as exc:
        if "planner_invalid_json" not in str(exc):
            raise
        repaired = request_ollama_chat(
            _planner_repair_system_prompt(),
            _planner_repair_user_prompt(question, raw),
            config,
        )
        data = parse_json_response(repaired, "planner")
    return legal_query_plan_from_json(data, question)


def legal_query_plan_from_json(data: dict[str, Any], question: str) -> LegalQueryPlan:
    allowed = {
        "intent",
        "question_scope",
        "normalized_question",
        "question_type",
        "answer_shape",
        "legal_terms",
        "legal_facets",
        "requested_components",
        "target_norm_roles",
        "entities",
        "target_doc_ids",
        "target_doc_aliases",
        "target_article_labels",
        "queries",
        "filters",
        "retrieval_bias",
        "needs_guidance_docs",
        "multi_hop_targets",
        "missing_facts",
        "confidence",
    }
    extra = sorted(set(data) - allowed)
    if extra:
        raise OllamaError(f"planner_unexpected_keys:{','.join(extra)}")

    intent = str(data.get("intent", "general"))
    if intent not in PLANNER_INTENTS:
        raise OllamaError(f"planner_invalid_intent:{intent}")

    queries = _parse_queries(data.get("queries"), question)
    doc_ids = _string_list(data.get("target_doc_ids"))
    article_labels = _string_list(data.get("target_article_labels"))
    inferred_doc_ids = extract_doc_ids(question)
    inferred_labels = extract_article_labels(question)
    _ensure_no_unsupported_doc_ids(doc_ids, inferred_doc_ids)
    _ensure_no_unsupported_article_labels(article_labels, inferred_labels)

    llm_guidance = bool(data.get("needs_guidance_docs"))
    rule_guidance = infer_needs_guidance(question, intent=intent)
    baseline = infer_runtime_metadata(
        question,
        intent=intent,
        target_doc_ids=doc_ids or inferred_doc_ids,
        target_article_labels=article_labels or inferred_labels,
        legal_terms=_string_list(data.get("legal_terms")),
        planned_queries=[query.text for query in queries],
        needs_guidance_docs=(llm_guidance or rule_guidance),
        confidence=_float_between_zero_and_one(data.get("confidence", 0.0)),
    )
    question_type = _pick_value(str(data.get("question_type") or ""), QUESTION_TYPES, baseline.question_type, "planner_invalid_question_type")
    answer_shape = _pick_value(str(data.get("answer_shape") or ""), ANSWER_SHAPES, baseline.answer_shape, "planner_invalid_answer_shape")
    retrieval_bias = _pick_value(
        str(data.get("retrieval_bias") or ""),
        RETRIEVAL_BIASES,
        baseline.retrieval_bias,
        "planner_invalid_retrieval_bias",
    )
    confidence = _float_between_zero_and_one(data.get("confidence", 0.0))
    legal_terms = _string_list(data.get("legal_terms")) or baseline.legal_facets[:]
    legal_facets = _string_list(data.get("legal_facets")) or baseline.legal_facets
    requested_components = _string_list(data.get("requested_components")) or baseline.requested_components
    target_norm_roles = _norm_roles(_string_list(data.get("target_norm_roles"))) or baseline.target_norm_roles
    filters = _filters(data.get("filters"))
    if not filters.get("doc_types"):
        filters["doc_types"] = infer_doc_type_hints(question)
    if not filters.get("must_include_terms"):
        filters["must_include_terms"] = infer_must_include_terms(question)
    domain_anchors = baseline.domain_anchors or infer_domain_anchors(question)
    governing_doc_hints = baseline.governing_doc_hints or infer_governing_doc_hints(question, domain_anchors=domain_anchors)
    if not filters.get("should_include_terms"):
        filters["should_include_terms"] = [*filters["must_include_terms"], *governing_doc_hints[:3], *legal_facets[:4]][:8]
    queries = _repair_queries(question, queries, baseline.legal_facets, question_type, domain_anchors, governing_doc_hints)
    multi_hop_targets = _string_list(data.get("multi_hop_targets"))
    if baseline.needs_guidance_docs and not multi_hop_targets:
        multi_hop_targets = ["Nghị định", "Thông tư"]
    return LegalQueryPlan(
        intent=intent,
        question_scope=str(data.get("question_scope") or "unknown"),
        normalized_question=str(data.get("normalized_question") or question).strip(),
        question_type=question_type,
        answer_shape=answer_shape,
        legal_terms=legal_terms,
        legal_facets=legal_facets,
        requested_components=requested_components,
        target_norm_roles=target_norm_roles,
        governing_doc_hints=governing_doc_hints,
        domain_anchors=domain_anchors,
        entities=_entities(data.get("entities")),
        target_doc_ids=doc_ids or inferred_doc_ids,
        target_doc_aliases=_string_list(data.get("target_doc_aliases")),
        target_article_labels=article_labels or inferred_labels,
        queries=queries,
        filters=filters,
        retrieval_bias=retrieval_bias,
        needs_guidance_docs=baseline.needs_guidance_docs,
        multi_hop_targets=multi_hop_targets,
        missing_facts=_string_list(data.get("missing_facts")),
        confidence=confidence,
    )


def fallback_plan_for_debug(question: str) -> LegalQueryPlan:
    labels = extract_article_labels(question)
    doc_ids = extract_doc_ids(question)
    baseline = infer_runtime_metadata(
        question,
        intent=_rule_intent(question),
        target_doc_ids=doc_ids,
        target_article_labels=labels,
        legal_terms=_important_terms(question),
        planned_queries=[question.strip(), " ".join(_important_terms(question)).strip()],
        needs_guidance_docs=_needs_guidance(question),
        confidence=0.35,
    )
    return LegalQueryPlan(
        intent=_rule_intent(question),
        question_scope="unknown",
        normalized_question=question.strip(),
        question_type=baseline.question_type,
        answer_shape=baseline.answer_shape,
        legal_terms=_important_terms(question),
        legal_facets=baseline.legal_facets,
        requested_components=baseline.requested_components,
        target_norm_roles=baseline.target_norm_roles,
        governing_doc_hints=baseline.governing_doc_hints,
        domain_anchors=baseline.domain_anchors,
        target_doc_ids=doc_ids,
        target_article_labels=labels,
        queries=_repair_queries(question, [
            PlannedQuery("original", question.strip(), "preserve user wording"),
            PlannedQuery("legal_terms", " ".join(_important_terms(question)), "BM25/exact legal terms"),
        ], baseline.legal_facets, baseline.question_type, baseline.domain_anchors, baseline.governing_doc_hints),
        filters={
            "doc_types": infer_doc_type_hints(question),
            "must_include_terms": infer_must_include_terms(question),
            "should_include_terms": [*infer_must_include_terms(question), *baseline.legal_facets[:4]][:6],
        },
        retrieval_bias=baseline.retrieval_bias,
        needs_guidance_docs=baseline.needs_guidance_docs,
        confidence=0.35,
    )


def _parse_queries(raw: Any, question: str) -> list[PlannedQuery]:
    if not isinstance(raw, list):
        raise OllamaError("planner_queries_not_list")
    output: list[PlannedQuery] = []
    seen: set[str] = set()
    for item in raw[:3]:
        if not isinstance(item, dict):
            raise OllamaError("planner_query_not_object")
        kind = str(item.get("kind", "")).strip()
        text = str(item.get("text", "")).strip()
        purpose = str(item.get("purpose", "")).strip()
        if kind not in QUERY_KINDS:
            raise OllamaError(f"planner_invalid_query_kind:{kind}")
        if not text:
            raise OllamaError("planner_empty_query")
        key = _query_key(text)
        if key in seen:
            continue
        seen.add(key)
        output.append(PlannedQuery(kind=kind, text=text, purpose=purpose))
    if not output:
        raise OllamaError("planner_no_queries")
    if all(query.text.strip().lower() != question.strip().lower() for query in output):
        output.insert(0, PlannedQuery("original", question.strip(), "preserve user wording"))
    return output[:3]


def _ensure_no_unsupported_doc_ids(doc_ids: list[str], inferred: list[str]) -> None:
    if not doc_ids:
        return
    allowed = {item.upper() for item in inferred}
    if allowed and all(doc_id.upper() in allowed for doc_id in doc_ids):
        return
    if not allowed:
        raise OllamaError("planner_doc_id_not_in_question")


def _ensure_no_unsupported_article_labels(labels: list[str], inferred: list[str]) -> None:
    if not labels:
        return
    allowed = {item.lower() for item in inferred}
    if allowed and all(label.lower() in allowed for label in labels):
        return
    if not allowed:
        raise OllamaError("planner_article_label_not_in_question")


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _entities(raw: Any) -> dict[str, list[str]]:
    keys = ["subjects", "actions", "objects", "conditions", "amounts_or_deadlines"]
    if not isinstance(raw, dict):
        return {key: [] for key in keys}
    return {key: _string_list(raw.get(key)) for key in keys}


def _filters(raw: Any) -> dict[str, list[str]]:
    keys = ["doc_types", "must_include_terms", "should_include_terms"]
    if not isinstance(raw, dict):
        return {key: [] for key in keys}
    return {key: _string_list(raw.get(key)) for key in keys}


def _norm_roles(values: list[str]) -> list[str]:
    return [value for value in values if value in NORM_ROLES]


def _float_between_zero_and_one(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _pick_value(raw: str, allowed: set[str], fallback: str, error_prefix: str) -> str:
    value = raw.strip()
    if not value:
        return fallback
    if value not in allowed:
        raise OllamaError(f"{error_prefix}:{value}")
    return value


def _repair_queries(
    question: str,
    queries: list[PlannedQuery],
    legal_facets: list[str],
    question_type: str,
    domain_anchors: list[str] | None = None,
    governing_doc_hints: list[str] | None = None,
) -> list[PlannedQuery]:
    output: list[PlannedQuery] = []
    seen: set[str] = set()

    def add(query: PlannedQuery) -> None:
        key = _query_key(query.text)
        if not key or key in seen or len(output) >= 3:
            return
        seen.add(key)
        output.append(query)

    for query in queries:
        add(query)
    if question_type == "list":
        for facet in legal_facets[:3]:
            if len(output) >= 3:
                break
            if facet.lower() in question.lower():
                add(PlannedQuery("expanded", f"{question.strip()} {facet}".strip(), "facet legal retrieval"))
            else:
                add(PlannedQuery("expanded", facet, "facet legal retrieval"))
    for hint in governing_doc_hints or []:
        if len(output) >= 3:
            break
        if hint.lower() not in question.lower():
            add(PlannedQuery("legal_terms", f"{question.strip()} {hint}".strip(), "governing legal anchor"))
    if not output and domain_anchors:
        add(PlannedQuery("legal_terms", " ".join(domain_anchors), "domain anchors"))
    if not output:
        add(PlannedQuery("original", question.strip(), "preserve user wording"))
    return output[:3]


def _needs_guidance(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in ["thủ tục", "hồ sơ", "xử phạt", "mức phạt", "thuế", "đất đai", "biểu mẫu"])


def _rule_intent(question: str) -> str:
    q = question.lower()
    if "xử phạt" in q or "mức phạt" in q:
        return "penalty"
    if "thủ tục" in q or "hồ sơ" in q:
        return "procedure"
    if "điều kiện" in q:
        return "condition"
    if "thuế" in q or "đất đai" in q:
        return "tax_land"
    if "hỗ trợ" in q or "ưu đãi" in q:
        return "support_policy"
    if "trách nhiệm" in q:
        return "responsibility"
    if "thẩm quyền" in q:
        return "authority"
    return "general"


def _important_terms(question: str) -> list[str]:
    terms = []
    q = question.lower()
    for term in [
        "doanh nghiệp nhỏ và vừa",
        "hỗ trợ",
        "ưu đãi",
        "thuế",
        "đất đai",
        "xử phạt",
        "hồ sơ",
        "thủ tục",
        "điều kiện",
        "trách nhiệm",
        "thẩm quyền",
        "Luật Thủ đô",
    ]:
        if term.lower() in q:
            terms.append(term)
    return terms or [question.strip()]


def _planner_system_prompt() -> str:
    return (
        "Bạn là Legal Query Planner cho RAG pháp luật Việt Nam. Không trả lời câu hỏi. "
        "Chỉ lập kế hoạch tra cứu từ chính câu hỏi; không bịa văn bản/số hiệu/Điều. "
        "Trả về một JSON object hợp lệ, không markdown. "
        "Giữ cụm pháp lý quan trọng: doanh nghiệp nhỏ và vừa, hỗ trợ, ưu đãi, thuế, đất đai, xử phạt, hồ sơ, thủ tục, "
        "điều kiện, trách nhiệm, thẩm quyền. "
        "Câu về thủ tục/hồ sơ/xử phạt/thuế/đất đai/hỗ trợ cụ thể đặt needs_guidance_docs=true. "
        "Câu nêu số hiệu hoặc Điều X thì đưa vào target_doc_ids/target_article_labels. "
        "Tạo tối đa 3 query ngắn, giàu thuật ngữ pháp lý."
    )


def _planner_repair_system_prompt() -> str:
    return (
        "Bạn sửa output của Legal Query Planner thành JSON hợp lệ duy nhất. "
        "Không thêm giải thích, không markdown. Giữ nguyên ý nghĩa, không bịa văn bản/Điều luật."
    )


def _planner_repair_user_prompt(question: str, raw: str) -> str:
    return f"""CÂU HỎI:
{question}

OUTPUT CẦN SỬA:
{raw[:6000]}

Trả lại duy nhất một JSON object hợp lệ theo schema planner. Không thêm text ngoài JSON."""


def _query_key(text: str) -> str:
    normalized = re.sub(r"[?？!！.,;:]+", "", text.strip().lower())
    return re.sub(r"\s+", " ", normalized)


def _planner_user_prompt(question: str) -> str:
    return f"""CÂU HỎI: {question}

Trả về JSON object ngắn đúng các key sau, không thêm key:
intent, question_type, answer_shape, legal_terms, legal_facets,
requested_components, target_norm_roles, queries, retrieval_bias,
needs_guidance_docs, confidence.

Allowed values:
intent=definition|condition|procedure|deadline|penalty|support_policy|tax_land|authority|responsibility|comparison|general
question_type=definition|list|condition|procedure|penalty|deadline|authority|comparison|mixed
answer_shape=single_rule|list_items|penalty_and_remedy|procedure_steps|conditions_list|document_pointer
retrieval_bias=content_articles|procedure_articles|sanction_articles|authority_articles
target_norm_roles values=procedure|authority|penalty|remedy|condition|support_policy
query.kind values=original|legal_terms|expanded

Required shapes:
queries=[{{"kind":"original","text":"...","purpose":"preserve user wording"}},{{"kind":"legal_terms","text":"...","purpose":"BM25/exact legal terms"}}]

Rules: confidence 0..1; max 3 non-duplicate queries; unknown doc ids/articles => empty arrays; list questions need facet queries."""
