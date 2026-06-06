from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from legal_rag.generation.ollama import OllamaConfig, OllamaError, parse_json_response, request_ollama_chat
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
    legal_terms: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    target_doc_ids: list[str] = field(default_factory=list)
    target_doc_aliases: list[str] = field(default_factory=list)
    target_article_labels: list[str] = field(default_factory=list)
    queries: list[PlannedQuery] = field(default_factory=list)
    filters: dict[str, list[str]] = field(default_factory=dict)
    needs_guidance_docs: bool = False
    multi_hop_targets: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["queries"] = [query.to_dict() for query in self.queries]
        return data


def plan_legal_query(question: str, config: OllamaConfig) -> LegalQueryPlan:
    raw = request_ollama_chat(_planner_system_prompt(), _planner_user_prompt(question), config)
    return legal_query_plan_from_json(parse_json_response(raw, "planner"), question)


def legal_query_plan_from_json(data: dict[str, Any], question: str) -> LegalQueryPlan:
    allowed = {
        "intent",
        "question_scope",
        "normalized_question",
        "legal_terms",
        "entities",
        "target_doc_ids",
        "target_doc_aliases",
        "target_article_labels",
        "queries",
        "filters",
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

    confidence = _float_between_zero_and_one(data.get("confidence", 0.0))
    return LegalQueryPlan(
        intent=intent,
        question_scope=str(data.get("question_scope") or "unknown"),
        normalized_question=str(data.get("normalized_question") or question).strip(),
        legal_terms=_string_list(data.get("legal_terms")),
        entities=_entities(data.get("entities")),
        target_doc_ids=doc_ids or inferred_doc_ids,
        target_doc_aliases=_string_list(data.get("target_doc_aliases")),
        target_article_labels=article_labels or inferred_labels,
        queries=queries,
        filters=_filters(data.get("filters")),
        needs_guidance_docs=bool(data.get("needs_guidance_docs")),
        multi_hop_targets=_string_list(data.get("multi_hop_targets")),
        missing_facts=_string_list(data.get("missing_facts")),
        confidence=confidence,
    )


def fallback_plan_for_debug(question: str) -> LegalQueryPlan:
    labels = extract_article_labels(question)
    doc_ids = extract_doc_ids(question)
    return LegalQueryPlan(
        intent=_rule_intent(question),
        question_scope="unknown",
        normalized_question=question.strip(),
        legal_terms=_important_terms(question),
        target_doc_ids=doc_ids,
        target_article_labels=labels,
        queries=[
            PlannedQuery("original", question.strip(), "preserve user wording"),
            PlannedQuery("legal_terms", " ".join(_important_terms(question)), "BM25/exact legal terms"),
        ],
        filters={"doc_types": [], "must_include_terms": [], "should_include_terms": []},
        needs_guidance_docs=_needs_guidance(question),
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
        key = re.sub(r"\s+", " ", text.lower())
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
    keys = ["subjects", "actions", "conditions", "amounts_or_deadlines"]
    if not isinstance(raw, dict):
        return {key: [] for key in keys}
    return {key: _string_list(raw.get(key)) for key in keys}


def _filters(raw: Any) -> dict[str, list[str]]:
    keys = ["doc_types", "must_include_terms", "should_include_terms"]
    if not isinstance(raw, dict):
        return {key: [] for key in keys}
    return {key: _string_list(raw.get(key)) for key in keys}


def _float_between_zero_and_one(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


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
        "Bạn là Legal Query Planner cho hệ thống RAG pháp luật Việt Nam.\n\n"
        "Nhiệm vụ của bạn KHÔNG phải trả lời câu hỏi. Nhiệm vụ là chuyển câu hỏi thành kế hoạch tra cứu pháp luật chính xác, "
        "dùng được cho BM25, dense retrieval, sparse retrieval và exact metadata filtering.\n\n"
        "Chỉ được dựa vào câu hỏi người dùng. Không bịa số hiệu văn bản, không bịa Điều luật, không bịa tên văn bản nếu câu hỏi "
        "không nêu hoặc không thể suy ra rất chắc. Trả về JSON hợp lệ duy nhất, không markdown, không giải thích ngoài JSON.\n\n"
        "Ưu tiên pháp luật Việt Nam:\n"
        '- Giữ nguyên cụm pháp lý quan trọng: "doanh nghiệp nhỏ và vừa", "hỗ trợ", "ưu đãi", "thuế", "đất đai", '
        '"xử phạt", "hồ sơ", "thủ tục", "điều kiện", "trách nhiệm", "thẩm quyền".\n'
        "- Nếu câu hỏi hỏi cách áp dụng, thủ tục, hồ sơ, mức phạt, thuế, đất đai, hỗ trợ cụ thể thì đánh dấu needs_guidance_docs=true.\n"
        "- Nếu câu hỏi nêu số hiệu văn bản hoặc Điều X thì đưa vào target_doc_ids/target_article_labels và tạo exact query.\n"
        "- Tạo tối đa 3 truy vấn: original, legal_terms, expanded. Truy vấn phải ngắn, giàu thuật ngữ pháp lý, không biến thành câu trả lời."
    )


def _planner_user_prompt(question: str) -> str:
    return f"""CÂU HỎI:
{question}

Trả về JSON theo schema sau:
{{
  "intent": "definition|condition|procedure|deadline|penalty|support_policy|tax_land|authority|responsibility|comparison|general",
  "question_scope": "single_article|multi_article_same_doc|multi_doc|unknown",
  "normalized_question": "...",
  "legal_terms": ["..."],
  "entities": {{
    "subjects": ["..."],
    "actions": ["..."],
    "conditions": ["..."],
    "amounts_or_deadlines": ["..."]
  }},
  "target_doc_ids": [],
  "target_doc_aliases": [],
  "target_article_labels": [],
  "queries": [
    {{"kind": "original", "text": "...", "purpose": "preserve user wording"}},
    {{"kind": "legal_terms", "text": "...", "purpose": "BM25/exact legal terms"}},
    {{"kind": "expanded", "text": "...", "purpose": "semantic retrieval with related legal wording"}}
  ],
  "filters": {{
    "doc_types": [],
    "must_include_terms": [],
    "should_include_terms": []
  }},
  "needs_guidance_docs": false,
  "multi_hop_targets": [],
  "missing_facts": [],
  "confidence": 0.0
}}

Ràng buộc:
- confidence từ 0 đến 1.
- queries tối đa 3, không trùng nhau.
- Nếu không chắc tên luật/số hiệu/Điều thì để mảng rỗng.
- Không thêm key ngoài schema."""
