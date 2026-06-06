from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from legal_rag.schemas.models import GoldQuestionMetadata, PredictedQuestionMetadata, Question
from legal_rag.utils.text import extract_article_labels, extract_doc_ids, normalize_text

LIST_MARKERS = (
    "những",
    "các",
    "trường hợp nào",
    "hình thức nào",
    "nội dung nào",
    "chính sách nào",
)
PROCEDURE_MARKERS = ("thủ tục", "hồ sơ", "trình tự", "cơ quan nào", "nộp ở đâu")
PENALTY_MARKERS = ("xử phạt", "mức phạt", "phạt", "khắc phục hậu quả")
DEADLINE_MARKERS = ("thời hạn", "bao lâu", "trong bao nhiêu ngày", "khi nào")
AUTHORITY_MARKERS = ("thẩm quyền", "cơ quan nào quyết định", "ai có quyền")
CONDITION_MARKERS = ("điều kiện", "tiêu chí", "khi nào", "trường hợp nào được")
COMPARISON_MARKERS = ("khác gì", "so với", "so sánh")
GUIDANCE_MARKERS = (
    "thủ tục",
    "hồ sơ",
    "biểu mẫu",
    "xử phạt",
    "mức phạt",
    "khắc phục hậu quả",
    "thuế",
    "hóa đơn",
    "mã số thuế",
    "đất đai",
    "thời hạn",
)
FACET_PATTERNS = [
    ("doanh nghiệp nhỏ và vừa", ("doanh nghiệp nhỏ và vừa", "sme")),
    ("hỗ trợ", ("hỗ trợ",)),
    ("ưu đãi", ("ưu đãi", "ưu đãi đầu tư")),
    ("thu hút đầu tư", ("thu hút đầu tư", "đầu tư", "phát triển doanh nghiệp")),
    ("khoa học công nghệ", ("khoa học", "công nghệ", "đổi mới sáng tạo", "chuyển đổi số")),
    ("nhân lực", ("nhân lực", "lao động", "nguồn nhân lực")),
    ("thuế", ("thuế", "mã số thuế", "kê khai thuế")),
    ("hóa đơn", ("hóa đơn",)),
    ("đất đai", ("đất đai", "đất",)),
    ("sở hữu trí tuệ", ("sở hữu trí tuệ", "nhãn hiệu", "sáng chế", "kiểu dáng", "tác giả")),
    ("xử phạt", ("xử phạt", "mức phạt", "vi phạm")),
    ("khắc phục hậu quả", ("khắc phục hậu quả",)),
    ("thủ tục hồ sơ", ("thủ tục", "hồ sơ", "trình tự", "biểu mẫu")),
    ("thẩm quyền", ("thẩm quyền", "cơ quan", "ủy ban", "bộ", "chính phủ")),
]
DOC_TYPE_HINTS = {
    "luật": "Luật",
    "nghị định": "Nghị định",
    "thông tư": "Thông tư",
    "nghị quyết": "Nghị quyết",
}


def infer_runtime_metadata(
    question: str,
    *,
    intent: str = "general",
    target_doc_ids: list[str] | None = None,
    target_article_labels: list[str] | None = None,
    legal_terms: list[str] | None = None,
    planned_queries: list[str] | None = None,
    needs_guidance_docs: bool | None = None,
    confidence: float = 0.0,
) -> PredictedQuestionMetadata:
    normalized = normalize_text(question)
    question_type = infer_question_type(normalized, intent=intent)
    answer_shape = infer_answer_shape(normalized, intent=intent, question_type=question_type)
    guidance = infer_needs_guidance(normalized, intent=intent) if needs_guidance_docs is None else needs_guidance_docs
    facets = infer_legal_facets(normalized, legal_terms=legal_terms)
    retrieval_bias = infer_retrieval_bias(intent=intent, question_type=question_type, answer_shape=answer_shape)
    return PredictedQuestionMetadata(
        intent=intent,
        question_type=question_type,
        answer_shape=answer_shape,
        needs_guidance_docs=guidance,
        target_doc_ids=target_doc_ids or extract_doc_ids(normalized),
        target_article_labels=target_article_labels or extract_article_labels(normalized),
        legal_facets=facets,
        retrieval_bias=retrieval_bias,
        planned_queries=[item for item in (planned_queries or []) if item.strip()],
        confidence=max(0.0, min(1.0, float(confidence))),
    )


def infer_question_type(question: str, *, intent: str = "general") -> str:
    lowered = question.lower()
    if any(marker in lowered for marker in PENALTY_MARKERS):
        return "penalty"
    if any(marker in lowered for marker in PROCEDURE_MARKERS):
        return "procedure"
    if any(marker in lowered for marker in DEADLINE_MARKERS):
        return "deadline"
    if any(marker in lowered for marker in AUTHORITY_MARKERS) or intent == "authority":
        return "authority"
    if any(marker in lowered for marker in COMPARISON_MARKERS) or intent == "comparison":
        return "comparison"
    if any(marker in lowered for marker in CONDITION_MARKERS) or intent == "condition":
        return "condition"
    if any(marker in lowered for marker in LIST_MARKERS):
        return "list"
    if lowered.startswith("theo điều ") or extract_article_labels(question):
        return "definition"
    return "mixed"


def infer_answer_shape(question: str, *, intent: str = "general", question_type: str = "mixed") -> str:
    lowered = question.lower()
    if question_type == "procedure":
        return "procedure_steps"
    if question_type == "penalty":
        if "khắc phục hậu quả" in lowered or "biện pháp khắc phục" in lowered:
            return "penalty_and_remedy"
        return "single_rule"
    if question_type == "list":
        return "list_items"
    if question_type == "condition":
        return "conditions_list"
    if question_type in {"authority", "deadline"}:
        return "document_pointer"
    if intent in {"support_policy", "tax_land"} and any(marker in lowered for marker in LIST_MARKERS):
        return "list_items"
    return "single_rule"


def infer_needs_guidance(question: str, *, intent: str = "general") -> bool:
    lowered = question.lower()
    if intent in {"procedure", "penalty", "tax_land"}:
        return True
    return any(marker in lowered for marker in GUIDANCE_MARKERS)


def infer_legal_facets(question: str, *, legal_terms: list[str] | None = None) -> list[str]:
    lowered = question.lower()
    facets: list[str] = []
    seen: set[str] = set()
    for label, patterns in FACET_PATTERNS:
        if any(pattern in lowered for pattern in patterns) and label not in seen:
            seen.add(label)
            facets.append(label)
    for term in legal_terms or []:
        normalized = term.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            facets.append(normalized)
    return facets[:8]


def infer_retrieval_bias(*, intent: str, question_type: str, answer_shape: str) -> str:
    if question_type == "procedure" or answer_shape == "procedure_steps":
        return "procedure_articles"
    if question_type == "penalty" or intent == "penalty":
        return "sanction_articles"
    if question_type == "authority" or intent in {"authority", "responsibility"}:
        return "authority_articles"
    return "content_articles"


def infer_doc_type_hints(question: str) -> list[str]:
    lowered = question.lower()
    return [doc_type for needle, doc_type in DOC_TYPE_HINTS.items() if needle in lowered]


def scaffold_gold_metadata(question: Question, predicted: PredictedQuestionMetadata) -> GoldQuestionMetadata:
    note_bits = [
        f"review_intent={predicted.intent}",
        f"review_type={predicted.question_type}",
        f"review_shape={predicted.answer_shape}",
    ]
    if predicted.legal_facets:
        note_bits.append(f"facets={','.join(predicted.legal_facets[:4])}")
    return GoldQuestionMetadata(
        question_id=question.id,
        intent=predicted.intent,
        question_type=predicted.question_type,
        answer_shape=predicted.answer_shape,
        needs_guidance_docs=predicted.needs_guidance_docs,
        gold_relevant_docs=[],
        gold_relevant_articles=[],
        notes="; ".join(note_bits),
    )


def split_questions_for_gold(
    questions: list[Question],
    *,
    holdout_ratio: float = 0.25,
    predicted_by_id: dict[int, PredictedQuestionMetadata] | None = None,
) -> tuple[list[Question], list[Question]]:
    predicted_by_id = predicted_by_id or {}
    buckets: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        predicted = predicted_by_id.get(question.id) or infer_runtime_metadata(question.question)
        key = f"{predicted.intent}|{predicted.question_type}|{predicted.answer_shape}|{int(predicted.needs_guidance_docs)}"
        buckets[key].append(question)

    tune: list[Question] = []
    holdout: list[Question] = []
    for bucket_key, items in buckets.items():
        ordered = sorted(items, key=lambda item: _stable_question_key(bucket_key, item))
        holdout_count = max(1, round(len(ordered) * holdout_ratio)) if len(ordered) > 2 else 1 if len(ordered) == 2 else 0
        holdout.extend(ordered[:holdout_count])
        tune.extend(ordered[holdout_count:])
    if not holdout and len(tune) > 1:
        tune.sort(key=lambda item: _stable_question_key("global", item))
        holdout.append(tune.pop(0))
    return sorted(tune, key=lambda item: item.id), sorted(holdout, key=lambda item: item.id)


def _stable_question_key(bucket_key: str, question: Question) -> str:
    digest = hashlib.sha256(f"{bucket_key}::{question.id}::{question.question}".encode("utf-8")).hexdigest()
    return digest
