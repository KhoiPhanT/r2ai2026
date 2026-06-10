from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from legal_rag.generation.ollama import OllamaConfig, OllamaError, parse_json_response, request_ollama_chat
from legal_rag.domain.components import COMPONENT_REGISTRY, detect_component_requirements
from legal_rag.lexicon import LegalLexiconMatch, normalize_legal_query_text
from legal_rag.question_metadata import (
    infer_component_requirements,
    infer_doc_type_hints,
    infer_domain_anchors,
    infer_governing_doc_hints,
    infer_must_include_terms,
    infer_needs_guidance,
    infer_requested_components,
    infer_runtime_metadata,
    infer_target_norm_roles,
)
from legal_rag.schemas.models import CorpusCandidate, MetadataSignal, PredictedQuestionMetadata, RequestedComponent
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
QUERY_KINDS = {"original", "exact", "core", "component", "guidance", "legal_terms", "expanded", "regime", "semantic"}
MAX_PLANNED_QUERIES = 3
QUESTION_TYPES = {"definition", "list", "condition", "procedure", "penalty", "deadline", "authority", "comparison", "mixed"}
ANSWER_SHAPES = {"single_rule", "list_items", "penalty_and_remedy", "procedure_steps", "conditions_list", "document_pointer"}
RETRIEVAL_BIASES = {"content_articles", "procedure_articles", "sanction_articles", "authority_articles"}
NORM_ROLES = {"procedure", "authority", "penalty", "remedy", "condition", "support_policy", "responsibility"}


@dataclass(slots=True)
class PlannedQuery:
    kind: str
    text: str
    purpose: str = ""
    source_signals: list[str] = field(default_factory=list)

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
    component_requirements: list[RequestedComponent] = field(default_factory=list)
    metadata_signals: list[MetadataSignal] = field(default_factory=list)
    reconciliation_issues: list[str] = field(default_factory=list)
    target_norm_roles: list[str] = field(default_factory=list)
    governing_doc_hints: list[str] = field(default_factory=list)
    domain_anchors: list[str] = field(default_factory=list)
    lexical_expansions: list[str] = field(default_factory=list)
    candidate_regimes: list[str] = field(default_factory=list)
    must_keep_phrases: list[str] = field(default_factory=list)
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
            lexical_expansions=self.lexical_expansions,
            candidate_regimes=self.candidate_regimes,
            must_keep_phrases=self.must_keep_phrases,
            planned_queries=[query.text for query in self.queries],
            signals=self.metadata_signals,
            component_requirements=self.component_requirements,
            reconciliation_issues=self.reconciliation_issues,
            confidence=self.confidence,
        )


def plan_legal_query(
    question: str,
    config: OllamaConfig,
    *,
    lexicon_candidates: list[LegalLexiconMatch] | None = None,
    corpus_candidates: list[CorpusCandidate] | None = None,
) -> LegalQueryPlan:
    grounded_lexicon = _grounded_lexicon_candidates(question, lexicon_candidates or [])
    runtime_config = planner_runtime_config(question, config)
    raw = request_ollama_chat(
        _planner_system_prompt(),
        _planner_user_prompt(question, grounded_lexicon, corpus_candidates or []),
        runtime_config,
        json_schema=_planner_json_schema(),
    )
    try:
        data = parse_json_response(raw, "planner")
    except OllamaError as exc:
        if "planner_invalid_json" not in str(exc):
            raise
        repair_config = replace(
            runtime_config,
            think=True,
            num_ctx=max(runtime_config.num_ctx, 12288),
            max_tokens=max(runtime_config.max_tokens, 520),
        )
        repaired = request_ollama_chat(
            _planner_repair_system_prompt(),
            _planner_repair_user_prompt(question, raw),
            repair_config,
            json_schema=_planner_json_schema(),
        )
        data = parse_json_response(repaired, "planner")
    return legal_query_plan_from_json(
        data,
        question,
        lexicon_candidates=grounded_lexicon,
        corpus_candidates=corpus_candidates,
    )


def planner_runtime_config(question: str, config: OllamaConfig) -> OllamaConfig:
    """Spend reasoning/context only where deterministic question signals show complexity."""
    metadata = infer_runtime_metadata(question)
    required_count = len(metadata.requested_components)
    word_count = len(question.split())
    hard_case = (
        metadata.question_type == "comparison"
        or required_count >= 3
        or word_count >= 70
    )
    if not hard_case:
        return config
    return replace(
        config,
        think=True,
        num_ctx=max(config.num_ctx, 12288),
        max_tokens=max(config.max_tokens, 380),
    )


def _planner_json_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": sorted(PLANNER_INTENTS)},
            "normalized_question": {"type": "string"},
            "question_type": {"type": "string", "enum": sorted(QUESTION_TYPES)},
            "answer_shape": {"type": "string", "enum": sorted(ANSWER_SHAPES)},
            "legal_terms": string_array,
            "requested_components": string_array,
            "entities": {
                "type": "object",
                "properties": {
                    "subjects": string_array,
                    "actions": string_array,
                    "objects": string_array,
                    "conditions": string_array,
                    "amounts_or_deadlines": string_array,
                },
                "required": ["subjects", "actions", "objects", "conditions", "amounts_or_deadlines"],
                "additionalProperties": False,
            },
            "target_doc_ids": string_array,
            "target_article_labels": string_array,
            "queries": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["exact", "core", "component", "guidance"]},
                        "text": {"type": "string"},
                        "purpose": {"type": "string"},
                    },
                    "required": ["kind", "text", "purpose"],
                    "additionalProperties": False,
                },
            },
            "retrieval_bias": {"type": "string", "enum": sorted(RETRIEVAL_BIASES)},
            "needs_guidance_docs": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "intent",
            "normalized_question",
            "question_type",
            "answer_shape",
            "legal_terms",
            "requested_components",
            "entities",
            "target_doc_ids",
            "target_article_labels",
            "queries",
            "retrieval_bias",
            "needs_guidance_docs",
            "confidence",
        ],
        "additionalProperties": False,
    }


def legal_query_plan_from_json(
    data: dict[str, Any],
    question: str,
    *,
    lexicon_candidates: list[LegalLexiconMatch] | None = None,
    corpus_candidates: list[CorpusCandidate] | None = None,
) -> LegalQueryPlan:
    lexicon_candidates = _grounded_lexicon_candidates(question, lexicon_candidates or [])
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
        "lexical_expansions",
        "candidate_regimes",
        "must_keep_phrases",
        "query_pack",
    }
    reconciliation_issues: list[str] = []
    extra = sorted(set(data) - allowed)
    if extra:
        reconciliation_issues.append(f"planner_extra_keys_dropped:{','.join(extra)}")
        data = {key: value for key, value in data.items() if key in allowed}

    intent = str(data.get("intent", "general"))
    if intent not in PLANNER_INTENTS:
        reconciliation_issues.append(f"planner_invalid_intent_sanitized:{intent}")
        intent = "general"

    queries = _parse_queries(data.get("query_pack") or data.get("queries"), question)
    proposed_doc_ids = _string_list(data.get("target_doc_ids"))
    article_labels = _string_list(data.get("target_article_labels"))
    target_doc_aliases = _string_list(data.get("target_doc_aliases"))
    entities = _entities(data.get("entities"))
    inferred_doc_ids = extract_doc_ids(question)
    inferred_labels = extract_article_labels(question)
    catalog_doc_ids = [item.doc_id for item in corpus_candidates or []]
    doc_ids, suggested_doc_ids, rejected_doc_ids = _partition_supported_doc_ids(
        proposed_doc_ids,
        inferred_doc_ids,
        catalog_doc_ids,
    )
    article_labels, suggested_article_labels = _partition_supported_article_labels(article_labels, inferred_labels)
    if suggested_doc_ids:
        reconciliation_issues.append(f"planner_doc_ids_catalog_grounded:{'|'.join(suggested_doc_ids)}")
    if rejected_doc_ids:
        reconciliation_issues.append(f"planner_doc_ids_rejected:{'|'.join(rejected_doc_ids)}")
    if suggested_article_labels:
        reconciliation_issues.append(f"planner_article_labels_demoted:{'|'.join(suggested_article_labels)}")

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
    llm_question_type = _pick_value(str(data.get("question_type") or ""), QUESTION_TYPES, baseline.question_type, "planner_invalid_question_type")
    question_type = baseline.question_type if baseline.question_type != "mixed" else llm_question_type
    llm_answer_shape = _pick_value(str(data.get("answer_shape") or ""), ANSWER_SHAPES, baseline.answer_shape, "planner_invalid_answer_shape")
    answer_shape = baseline.answer_shape if baseline.answer_shape != "single_rule" else llm_answer_shape
    retrieval_bias = _pick_value(
        str(data.get("retrieval_bias") or ""),
        RETRIEVAL_BIASES,
        baseline.retrieval_bias,
        "planner_invalid_retrieval_bias",
    )
    confidence = _float_between_zero_and_one(data.get("confidence", 0.0))
    lexicon_terms = _lexicon_terms(lexicon_candidates or [])
    lexicon_regimes = [
        regime
        for regime in _lexicon_regimes(lexicon_candidates or [])
        if _question_mentions_local_scope(question) or not _looks_like_local_regime(regime)
    ]
    lexicon_doc_hints = _lexicon_doc_hints(lexicon_candidates or [])
    catalog_hints = _selected_catalog_hints(suggested_doc_ids, corpus_candidates or [])
    legal_terms = _merge_unique(_string_list(data.get("legal_terms")), lexicon_terms[:6]) or baseline.legal_facets[:]
    legal_facets = _merge_unique(baseline.legal_facets, lexicon_terms[:8])
    component_requirements = _reconcile_component_requirements(
        question,
        baseline.component_requirements,
        _string_list(data.get("requested_components")),
    )
    requested_components = [item.name for item in component_requirements if item.requirement == "required"]
    target_norm_roles = baseline.target_norm_roles
    filters = _filters(data.get("filters"))
    if not filters.get("doc_types"):
        filters["doc_types"] = infer_doc_type_hints(question)
    if not filters.get("must_include_terms"):
        filters["must_include_terms"] = infer_must_include_terms(question)
    domain_anchors = baseline.domain_anchors or infer_domain_anchors(question)
    governing_doc_hints = _merge_unique(
        baseline.governing_doc_hints or infer_governing_doc_hints(question, domain_anchors=domain_anchors),
        suggested_doc_ids,
        catalog_hints,
        lexicon_doc_hints[:6],
        lexicon_regimes[:4],
    )
    lexical_expansions = _merge_unique(_string_list(data.get("lexical_expansions")), lexicon_terms[:10])
    candidate_regimes = _merge_unique(_string_list(data.get("candidate_regimes")), lexicon_regimes[:8], lexicon_doc_hints[:6])
    must_keep_phrases = _merge_unique(_string_list(data.get("must_keep_phrases")), _must_keep_from_question(question))
    if not filters.get("should_include_terms"):
        filters["should_include_terms"] = [*filters["must_include_terms"], *governing_doc_hints[:3], *legal_facets[:4]][:8]
    queries = _repair_queries(
        question,
        queries,
        baseline.legal_facets,
        question_type,
        domain_anchors,
        governing_doc_hints,
        legal_terms=legal_terms,
        requested_components=requested_components,
        target_norm_roles=target_norm_roles,
        lexical_expansions=lexical_expansions,
        candidate_regimes=candidate_regimes,
        must_keep_phrases=must_keep_phrases,
        target_doc_ids=doc_ids or inferred_doc_ids,
        target_doc_aliases=target_doc_aliases,
        target_article_labels=article_labels or inferred_labels,
        entities=entities,
        needs_guidance_docs=baseline.needs_guidance_docs,
    )
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
        component_requirements=component_requirements,
        metadata_signals=_merge_metadata_signals(
            baseline.signals,
            lexicon_candidates or [],
            suggested_doc_ids,
            rejected_doc_ids,
        ),
        reconciliation_issues=reconciliation_issues,
        target_norm_roles=target_norm_roles,
        governing_doc_hints=governing_doc_hints,
        domain_anchors=domain_anchors,
        lexical_expansions=lexical_expansions,
        candidate_regimes=candidate_regimes,
        must_keep_phrases=must_keep_phrases,
        entities=entities,
        target_doc_ids=doc_ids or inferred_doc_ids,
        target_doc_aliases=target_doc_aliases,
        target_article_labels=article_labels or inferred_labels,
        queries=queries,
        filters=filters,
        retrieval_bias=retrieval_bias,
        needs_guidance_docs=baseline.needs_guidance_docs,
        multi_hop_targets=multi_hop_targets,
        missing_facts=_string_list(data.get("missing_facts")),
        confidence=confidence,
    )


def fallback_plan_for_debug(question: str, *, lexicon_candidates: list[LegalLexiconMatch] | None = None) -> LegalQueryPlan:
    lexicon_candidates = _grounded_lexicon_candidates(question, lexicon_candidates or [])
    labels = extract_article_labels(question)
    doc_ids = extract_doc_ids(question)
    lexicon_terms = _lexicon_terms(lexicon_candidates or [])
    lexicon_regimes = _lexicon_regimes(lexicon_candidates or [])
    lexicon_doc_hints = _lexicon_doc_hints(lexicon_candidates or [])
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
        legal_terms=_merge_unique(_important_terms(question), lexicon_terms[:6]),
        legal_facets=_merge_unique(baseline.legal_facets, lexicon_terms[:8]),
        requested_components=baseline.requested_components,
        component_requirements=baseline.component_requirements,
        metadata_signals=_merge_metadata_signals(baseline.signals, lexicon_candidates or [], []),
        target_norm_roles=baseline.target_norm_roles,
        governing_doc_hints=_merge_unique(baseline.governing_doc_hints, lexicon_doc_hints[:6], lexicon_regimes[:4]),
        domain_anchors=baseline.domain_anchors,
        lexical_expansions=lexicon_terms[:10],
        candidate_regimes=_merge_unique(lexicon_regimes[:8], lexicon_doc_hints[:6]),
        must_keep_phrases=_merge_unique(_must_keep_from_question(question), lexicon_terms[:6]),
        target_doc_ids=doc_ids,
        target_article_labels=labels,
        queries=_repair_queries(
            question,
            [
                PlannedQuery("original", question.strip(), "preserve user wording"),
                PlannedQuery("legal_terms", " ".join(_important_terms(question)), "BM25/exact legal terms"),
            ],
            baseline.legal_facets,
            baseline.question_type,
            baseline.domain_anchors,
            _merge_unique(baseline.governing_doc_hints, lexicon_doc_hints[:6], lexicon_regimes[:4]),
            legal_terms=_merge_unique(_important_terms(question), lexicon_terms[:6]),
            requested_components=baseline.requested_components,
            target_norm_roles=baseline.target_norm_roles,
            lexical_expansions=lexicon_terms[:10],
            candidate_regimes=_merge_unique(lexicon_regimes[:8], lexicon_doc_hints[:6]),
            must_keep_phrases=_merge_unique(_must_keep_from_question(question), lexicon_terms[:6]),
            target_doc_ids=doc_ids,
            target_article_labels=labels,
            entities={"subjects": baseline.legal_subjects, "actions": baseline.legal_actions, "objects": baseline.legal_objects},
            needs_guidance_docs=baseline.needs_guidance_docs,
        ),
        filters={
            "doc_types": infer_doc_type_hints(question),
            "must_include_terms": infer_must_include_terms(question),
            "should_include_terms": [*infer_must_include_terms(question), *baseline.legal_facets[:4], *lexicon_terms[:4]][:8],
        },
        retrieval_bias=baseline.retrieval_bias,
        needs_guidance_docs=baseline.needs_guidance_docs,
        confidence=0.35,
    )


def _parse_queries(raw: Any, question: str) -> list[PlannedQuery]:
    if not isinstance(raw, list):
        return [PlannedQuery("core", _fallback_query_text(question), "deterministic fallback", ["question"])]
    output: list[PlannedQuery] = []
    seen: set[str] = set()
    for item in raw[:MAX_PLANNED_QUERIES]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip()
        text = str(item.get("text", "")).strip()
        purpose = str(item.get("purpose", "")).strip()
        if kind not in QUERY_KINDS:
            kind = "core"
        if not text:
            continue
        key = _query_key(text)
        if key in seen:
            continue
        seen.add(key)
        output.append(PlannedQuery(kind=kind, text=text, purpose=purpose, source_signals=["question_llm"]))
    if not output:
        output.append(PlannedQuery("core", _fallback_query_text(question), "deterministic fallback", ["question"]))
    return output[:MAX_PLANNED_QUERIES]


def _partition_supported_doc_ids(
    doc_ids: list[str],
    inferred: list[str],
    catalog_doc_ids: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    allowed = {item.upper() for item in inferred}
    catalog = {item.upper() for item in (catalog_doc_ids or [])}
    supported = [item for item in doc_ids if item.upper() in allowed]
    suggested = [item for item in doc_ids if item.upper() not in allowed and item.upper() in catalog]
    rejected = [item for item in doc_ids if item.upper() not in allowed and item.upper() not in catalog]
    return supported, suggested, rejected


def _partition_supported_article_labels(labels: list[str], inferred: list[str]) -> tuple[list[str], list[str]]:
    allowed = {item.lower() for item in inferred}
    supported = [item for item in labels if item.lower() in allowed]
    suggested = [item for item in labels if item.lower() not in allowed]
    return supported, suggested


def _reconcile_component_requirements(
    question: str,
    baseline: list[RequestedComponent],
    llm_components: list[str],
) -> list[RequestedComponent]:
    output = list(baseline)
    known = {item.name for item in output}
    lowered = question.lower()
    for name in llm_components:
        if name in known:
            continue
        source_span = name if name.lower() in lowered else ""
        registered = name.lower() in COMPONENT_REGISTRY
        output.append(
            RequestedComponent(
                name=name,
                requirement="required" if source_span and registered else "optional",
                source_span=source_span,
                confidence=0.82 if source_span and registered else 0.5,
                source="question_llm",
            )
        )
        known.add(name)
    return output


def _merge_metadata_signals(
    baseline: list[MetadataSignal],
    lexicon_candidates: list[LegalLexiconMatch],
    suggested_doc_ids: list[str],
    rejected_doc_ids: list[str] | None = None,
) -> list[MetadataSignal]:
    output = list(baseline)
    for candidate in [item for item in lexicon_candidates if item.score >= 3.0][:4]:
        output.append(
            MetadataSignal(
                "lexicon_term",
                candidate.term,
                "lexicon",
                candidate.term,
                min(0.95, max(0.0, candidate.score / 7.0)),
                "advisory",
            )
        )
    for doc_id in suggested_doc_ids:
        output.append(MetadataSignal("suggested_doc_id", doc_id, "corpus_catalog", doc_id, 0.65, "advisory"))
    for doc_id in rejected_doc_ids or []:
        output.append(MetadataSignal("rejected_doc_id", doc_id, "question_llm", doc_id, 0.0, "advisory"))
    return output


def _selected_catalog_hints(doc_ids: list[str], candidates: list[CorpusCandidate]) -> list[str]:
    selected = {item.upper() for item in doc_ids}
    output: list[str] = []
    for candidate in candidates:
        if candidate.doc_id.upper() not in selected:
            continue
        output.extend([candidate.doc_id, candidate.title])
    return _merge_unique(output)[:6]


def _fallback_query_text(question: str) -> str:
    terms = _important_terms(question)
    return " ".join(terms) if terms else question.strip()


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).replace("_", " ").strip() for item in raw if str(item).strip()]


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


def _merge_unique(*groups: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                output.append(value)
    return output


def _question_supported_values(values: list[str], question: str) -> list[str]:
    lowered = question.lower()
    return [value for value in values if value.lower() in lowered]


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
        return fallback
    return value


def _repair_queries(
    question: str,
    queries: list[PlannedQuery],
    legal_facets: list[str],
    question_type: str,
    domain_anchors: list[str] | None = None,
    governing_doc_hints: list[str] | None = None,
    *,
    legal_terms: list[str] | None = None,
    requested_components: list[str] | None = None,
    target_norm_roles: list[str] | None = None,
    target_doc_ids: list[str] | None = None,
    target_doc_aliases: list[str] | None = None,
    target_article_labels: list[str] | None = None,
    entities: dict[str, list[str]] | None = None,
    needs_guidance_docs: bool = False,
    lexical_expansions: list[str] | None = None,
    candidate_regimes: list[str] | None = None,
    must_keep_phrases: list[str] | None = None,
) -> list[PlannedQuery]:
    output: list[PlannedQuery] = []
    seen: set[str] = set()

    def add(query: PlannedQuery) -> None:
        compacted = _compact_query_text(query.text)
        if not compacted:
            return
        key = _query_key(compacted)
        if not key or key in seen or len(output) >= MAX_PLANNED_QUERIES:
            return
        seen.add(key)
        output.append(PlannedQuery(query.kind, compacted, query.purpose, query.source_signals))

    entities = entities or {}
    legal_terms = legal_terms or []
    requested_components = requested_components or []
    target_norm_roles = target_norm_roles or []
    target_doc_ids = target_doc_ids or []
    target_doc_aliases = target_doc_aliases or []
    target_article_labels = target_article_labels or []
    governing_doc_hints = governing_doc_hints or []
    lexical_expansions = lexical_expansions or []
    candidate_regimes = candidate_regimes or []
    must_keep_phrases = must_keep_phrases or []

    exact_phrases = [*target_doc_ids, *target_doc_aliases, *target_article_labels]
    if exact_phrases:
        add(PlannedQuery("exact", _join_query_phrases(exact_phrases, max_words=14), "exact document/article anchor", ["explicit_target"]))

    add(
        PlannedQuery(
            "original",
            _compact_query_text(question, max_words=24),
            "preserve the user's legal wording",
            ["question"],
        )
    )

    llm_by_role = _llm_queries_by_role(queries)
    core_phrases = [
        *must_keep_phrases[:3],
        # Specific corpus-derived phrases should survive the word budget ahead
        # of broad planner labels such as "hỗ trợ" or "điều kiện".
        *lexical_expansions[:4],
        *entities.get("subjects", [])[:2],
        *entities.get("actions", [])[:2],
        *entities.get("objects", [])[:2],
        *requested_components[:2],
        *legal_terms[:2],
    ]
    add(
        PlannedQuery(
            "core",
            _join_query_phrases(core_phrases or [question], max_words=18),
            "lexical legal frame",
            ["question_rule", "legal_frame", "lexicon"],
        )
    )

    if len(requested_components) > 1 and len(output) < MAX_PLANNED_QUERIES:
        component_phrases = [
            *requested_components[:4],
            *must_keep_phrases[:3],
            *lexical_expansions[:2],
            *legal_facets[:2],
            *entities.get("subjects", [])[:2],
            *entities.get("objects", [])[:2],
        ]
        add(
            PlannedQuery(
                "component",
                _join_query_phrases(component_phrases or legal_facets or legal_terms, max_words=18),
                "required component coverage",
                ["required_component", "lexicon"],
            )
        )

    for query in [*llm_by_role.get("core", [])[:1], *llm_by_role.get("component", [])[:1]]:
        add(PlannedQuery("semantic", query.text, query.purpose or "semantic legal rewrite", ["question_llm"]))

    if not output and domain_anchors:
        add(PlannedQuery("core", " ".join(domain_anchors), "domain anchors", ["question_rule"]))
    if not output:
        add(PlannedQuery("original", question.strip(), "preserve user wording"))
    return output[:MAX_PLANNED_QUERIES]


def _llm_queries_by_role(queries: list[PlannedQuery]) -> dict[str, list[PlannedQuery]]:
    output: dict[str, list[PlannedQuery]] = {}
    for query in queries:
        role = {"legal_terms": "core", "expanded": "component", "regime": "core", "semantic": "component"}.get(query.kind, query.kind)
        output.setdefault(role, []).append(PlannedQuery(role, query.text, query.purpose, query.source_signals))
    return output


def _lexicon_terms(candidates: list[LegalLexiconMatch]) -> list[str]:
    output: list[str] = []
    for item in candidates:
        if item.score < 3.0:
            continue
        value = " ".join(str(item.term).split()).strip()
        if value and value.lower() not in {seen.lower() for seen in output}:
            output.append(value)
    for item in candidates:
        if item.score < 3.0:
            continue
        for value in item.aliases[:2]:
            value = " ".join(str(value).split()).strip()
            if value and value.lower() not in {seen.lower() for seen in output}:
                output.append(value)
    return output[:16]


def _grounded_lexicon_candidates(
    question: str,
    candidates: list[LegalLexiconMatch],
) -> list[LegalLexiconMatch]:
    normalized_question = normalize_legal_query_text(question)
    question_tokens = _content_tokens(normalized_question)
    question_bigrams = set(zip(question_tokens, question_tokens[1:]))
    output: list[LegalLexiconMatch] = []
    for item in candidates:
        surfaces = [item.term, *item.aliases[:3]]
        grounded = False
        for surface in surfaces:
            normalized_surface = normalize_legal_query_text(surface)
            if normalized_surface and normalized_surface in normalized_question:
                grounded = True
                break
            surface_tokens = _content_tokens(normalized_surface)
            if len(surface_tokens) < 3:
                continue
            surface_bigrams = list(zip(surface_tokens, surface_tokens[1:]))
            overlap = sum(1 for bigram in surface_bigrams if bigram in question_bigrams)
            if overlap >= 2 and overlap / len(surface_bigrams) >= 0.5:
                grounded = True
                break
        if grounded:
            output.append(item)
    return output


def _content_tokens(text: str) -> list[str]:
    generic = {
        "các",
        "công",
        "công ty",
        "của",
        "doanh nghiệp",
        "được",
        "gì",
        "khi",
        "những",
        "quy định",
        "theo",
        "thông tin",
        "tài liệu",
        "và",
        "về",
    }
    tokens = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in generic]


def _lexicon_regimes(candidates: list[LegalLexiconMatch]) -> list[str]:
    output: list[str] = []
    for item in candidates:
        if item.score < 3.0:
            continue
        for value in [*item.regimes[:2], item.term]:
            if any(marker in value.lower() for marker in ("luật", "nghị định", "thông tư", "hỗ trợ", "xử phạt", "thuế")):
                if value not in output:
                    output.append(value)
    return output[:12]


def _lexicon_doc_hints(candidates: list[LegalLexiconMatch]) -> list[str]:
    output: list[str] = []
    for item in candidates:
        if not re.fullmatch(r"\d{1,4}/\d{4}/[A-ZĐ\-]+(?:\d+)?", item.term, flags=re.IGNORECASE):
            continue
        for doc_id in item.doc_ids[:4]:
            if doc_id and doc_id not in output:
                output.append(doc_id)
    return output[:10]


def _must_keep_from_question(question: str) -> list[str]:
    phrases: list[str] = []
    for component in detect_component_requirements(question):
        if component.source_span and component.source_span not in phrases:
            phrases.append(component.source_span)
    for value in [*extract_doc_ids(question), *extract_article_labels(question)]:
        if value not in phrases:
            phrases.append(value)
    for quoted in re.findall(r'["“”](.{2,100}?)["“”]', question):
        normalized = " ".join(quoted.split())
        if normalized and normalized not in phrases:
            phrases.append(normalized)
    return phrases[:8]


def _guidance_hint_phrases(governing_doc_hints: list[str]) -> list[str]:
    if len(governing_doc_hints) >= 2 and governing_doc_hints[0].lower() in governing_doc_hints[1].lower():
        return [governing_doc_hints[1]]
    return governing_doc_hints[:2]


def _join_query_phrases(phrases: list[str], *, max_words: int) -> str:
    output: list[str] = []
    seen: set[str] = set()
    words = 0
    for phrase in phrases:
        normalized = _compact_query_text(phrase)
        key = _query_key(normalized)
        if not key or key in seen:
            continue
        phrase_words = normalized.split()
        if output and words + len(phrase_words) > max_words:
            continue
        seen.add(key)
        output.append(normalized)
        words += len(phrase_words)
    return " ".join(output).strip()


def _compact_query_text(text: str, *, max_words: int = 22) -> str:
    normalized = normalize_legal_query_text(text.replace("_", " ").replace("\n", " "))
    normalized = re.sub(r"\s+", " ", normalized).strip(" \t\r\n?.!,;:")
    words = normalized.split()
    if len(words) <= max_words:
        return normalized
    return " ".join(words[:max_words]).strip(" \t\r\n?.!,;:")


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
    terms = [item.source_span for item in detect_component_requirements(question) if item.source_span]
    terms.extend(extract_doc_ids(question))
    terms.extend(extract_article_labels(question))
    return _merge_unique(terms, [normalize_legal_query_text(question)])[:6]


def _planner_system_prompt() -> str:
    return (
        "Bạn là bộ phân tích câu hỏi và chiến lược truy hồi pháp luật Việt Nam, không phải bộ trả lời. "
        "Tách đúng chủ thể, hành vi, đối tượng, điều kiện và thành phần người dùng trực tiếp yêu cầu. "
        "Không suy một component chỉ vì từ đó xuất hiện trong bối cảnh khác: tiền phạt nộp thừa không phải hỏi mức phạt; "
        "trách nhiệm không đồng nghĩa thẩm quyền; câu dạng danh sách không mặc nhiên là chính sách hỗ trợ. "
        "Chỉ ghi doc_id hoặc Điều khi chúng xuất hiện nguyên văn trong câu hỏi. "
        "Query ngắn theo cấu trúc chủ thể + hành vi/vấn đề + component, ưu tiên thuật ngữ phù hợp từ lexicon. "
        "Trả duy nhất JSON đúng schema, không markdown và không giải thích."
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


def _planner_user_prompt(
    question: str,
    lexicon_candidates: list[LegalLexiconMatch] | None = None,
    corpus_candidates: list[CorpusCandidate] | None = None,
) -> str:
    candidate_lines = []
    strong_candidates = [item for item in (lexicon_candidates or []) if item.score >= 3.0][:4]
    for idx, item in enumerate(strong_candidates, start=1):
        bits = [item.term, *item.aliases[:2], *item.regimes[:1]]
        candidate_lines.append(f"L{idx}: " + " | ".join(bit for bit in bits if bit))
    candidate_block = "\n".join(candidate_lines) if candidate_lines else "(không có)"
    corpus_lines = []
    for idx, item in enumerate((corpus_candidates or [])[:8], start=1):
        labels = ", ".join(item.article_labels[:3]) or "không rõ Điều"
        corpus_lines.append(f"C{idx}: {item.doc_id} | {item.title} | {labels}")
    corpus_block = "\n".join(corpus_lines) if corpus_lines else "(không có)"
    return f"""CÂU HỎI:
{question}

LEXICON_CANDIDATES từ corpus:
{candidate_block}

CORPUS_CANDIDATES đã được tra từ kho văn bản thực tế:
{corpus_block}

Trả về JSON compact với đúng các key:
intent, normalized_question, question_type, answer_shape, legal_terms, requested_components, entities,
target_doc_ids, target_article_labels, queries, retrieval_bias, needs_guidance_docs, confidence.

Enum:
intent=definition|condition|procedure|deadline|penalty|support_policy|tax_land|authority|responsibility|comparison|general
question_type=definition|list|condition|procedure|penalty|deadline|authority|comparison|mixed
answer_shape=single_rule|list_items|penalty_and_remedy|procedure_steps|conditions_list|document_pointer
retrieval_bias=content_articles|procedure_articles|sanction_articles|authority_articles
query.kind=exact|core|component|guidance

Schema:
{{"intent":"...","normalized_question":"...","question_type":"mixed","answer_shape":"single_rule","legal_terms":[],"requested_components":[],"entities":{{"subjects":[],"actions":[],"objects":[],"conditions":[],"amounts_or_deadlines":[]}},"target_doc_ids":[],"target_article_labels":[],"queries":[{{"kind":"core","text":"...","purpose":"core legal issue"}}],"retrieval_bias":"content_articles","needs_guidance_docs":false,"confidence":0.0}}

Luật query: tối đa 2 query, không trùng, không dùng dấu gạch dưới.
- core: 8-16 từ pháp lý, giữ cụm quan trọng.
- component: chỉ khi câu hỏi nhiều ý/list/thủ tục/phạt.
- guidance: chỉ khi cần nghị định/thông tư/hướng dẫn.
- exact: chỉ khi câu hỏi nêu số hiệu/tên luật/Điều rõ.
- target_doc_ids chỉ được lấy từ câu hỏi hoặc đúng một doc_id trong CORPUS_CANDIDATES. Nếu chưa chắc, để rỗng.
- Không đổi hành vi pháp lý thành khái niệm gần nghĩa: "cho thuê doanh nghiệp" phải giữ "cho thuê", không đổi thành "chuyển giao".
- requested_components chỉ chứa phần được hỏi trực tiếp; không thêm phần chỉ có thể hữu ích.
- Ưu tiên dùng thuật ngữ trong LEXICON_CANDIDATES nếu phù hợp với câu hỏi.
- Nếu câu hỏi dùng từ đời thường, chuyển sang thuật ngữ pháp lý gần nhất từ corpus.
Không bịa doc_id/Điều. Không giải thích."""


def _question_mentions_local_scope(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("địa bàn", "tỉnh", "thành phố", "hđnd", "ubnd", "thủ đô"))


def _looks_like_local_regime(value: str) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in ("hội đồng nhân dân", "hđnd", "địa bàn tỉnh", "địa bàn thành phố"))
