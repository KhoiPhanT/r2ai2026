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
PENALTY_MARKERS = ("xử phạt", "mức phạt", "phạt", "khắc phục hậu quả", "bị xử lý", "xử lý như thế nào")
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
    ("cơ sở ươm tạo", ("cơ sở ươm tạo", "ươm tạo")),
    ("khu làm việc chung", ("khu làm việc chung",)),
    ("hỗ trợ", ("hỗ trợ",)),
    ("ưu đãi", ("ưu đãi", "ưu đãi đầu tư")),
    ("thu hút đầu tư", ("thu hút đầu tư", "đầu tư", "phát triển doanh nghiệp")),
    ("khoa học công nghệ", ("khoa học", "công nghệ", "đổi mới sáng tạo", "chuyển đổi số")),
    ("nhân lực", ("nhân lực", "lao động", "nguồn nhân lực")),
    ("thuế", ("thuế", "mã số thuế", "kê khai thuế")),
    ("hóa đơn", ("hóa đơn",)),
    ("đất đai", ("đất đai", "đất",)),
    ("đấu thầu", ("đấu thầu", "nhà thầu")),
    ("bảo hiểm xã hội", ("bảo hiểm xã hội", "bhxh")),
    ("bảo hiểm thất nghiệp", ("bảo hiểm thất nghiệp",)),
    ("hợp đồng lao động", ("hợp đồng lao động",)),
    ("bằng cấp", ("bằng cấp", "văn bằng", "chứng chỉ")),
    ("sở hữu trí tuệ", ("sở hữu trí tuệ", "nhãn hiệu", "sáng chế", "kiểu dáng", "tác giả")),
    ("xử phạt", ("xử phạt", "mức phạt", "vi phạm")),
    ("khắc phục hậu quả", ("khắc phục hậu quả",)),
    ("thủ tục hồ sơ", ("thủ tục", "hồ sơ", "trình tự", "biểu mẫu")),
    ("thẩm quyền", ("thẩm quyền", "cơ quan", "ủy ban", "bộ", "chính phủ")),
]
PROTECTED_TERM_GROUPS = (
    ("doanh nghiệp nhỏ và vừa", ("doanh nghiệp nhỏ và vừa", "sme")),
    ("cơ sở ươm tạo", ("cơ sở ươm tạo", "ươm tạo")),
    ("khu làm việc chung", ("khu làm việc chung",)),
    ("khởi nghiệp sáng tạo", ("khởi nghiệp sáng tạo",)),
    ("chuỗi giá trị", ("chuỗi giá trị",)),
    ("quỹ phát triển doanh nghiệp nhỏ và vừa", ("quỹ phát triển doanh nghiệp nhỏ và vừa",)),
    ("thuế", ("thuế", "ưu đãi thuế", "miễn thuế")),
    ("đất đai", ("đất đai", "đất")),
    ("hóa đơn", ("hóa đơn",)),
    ("mã số thuế", ("mã số thuế",)),
    ("hợp đồng lao động", ("hợp đồng lao động", "người lao động", "người sử dụng lao động")),
    ("bảo hiểm xã hội", ("bảo hiểm xã hội", "bhxh")),
    ("bảo hiểm thất nghiệp", ("bảo hiểm thất nghiệp",)),
    ("bằng cấp", ("bằng cấp", "văn bằng", "chứng chỉ")),
    ("sở hữu trí tuệ", ("sở hữu trí tuệ", "nhãn hiệu", "sáng chế", "kiểu dáng")),
    ("đấu thầu", ("đấu thầu", "nhà thầu")),
)
DOC_TYPE_HINTS = {
    "luật": "Luật",
    "bộ luật": "Bộ luật",
    "nghị định": "Nghị định",
    "thông tư": "Thông tư",
    "nghị quyết": "Nghị quyết",
}
DOMAIN_ANCHOR_RULES = (
    {
        "anchor": "support_policy_sme",
        "patterns": (
            "doanh nghiệp nhỏ và vừa",
            "doanh nghiệp siêu nhỏ",
            "cơ sở ươm tạo",
            "khu làm việc chung",
            "khởi nghiệp sáng tạo",
            "chuỗi giá trị",
        ),
        "governing_hints": (
            "hỗ trợ doanh nghiệp nhỏ và vừa",
            "nghị định hướng dẫn hỗ trợ doanh nghiệp nhỏ và vừa",
        ),
    },
    {
        "anchor": "labor_sanctions",
        "patterns": (
            "hợp đồng lao động",
            "người lao động",
            "người sử dụng lao động",
            "bằng cấp",
            "văn bằng",
            "chứng chỉ",
        ),
        "governing_hints": (
            "bộ luật lao động",
            "xử phạt vi phạm hành chính lao động",
        ),
    },
    {
        "anchor": "tax_admin_penalties",
        "patterns": (
            "thuế",
            "mã số thuế",
            "hóa đơn",
            "quản lý thuế",
            "cưỡng chế",
        ),
        "governing_hints": (
            "luật quản lý thuế",
            "xử phạt vi phạm hành chính thuế",
            "xử phạt vi phạm hành chính hóa đơn",
        ),
    },
    {
        "anchor": "intellectual_property",
        "patterns": (
            "sở hữu trí tuệ",
            "nhãn hiệu",
            "sáng chế",
            "kiểu dáng",
            "quyền tác giả",
        ),
        "governing_hints": (
            "luật sở hữu trí tuệ",
            "xử phạt sở hữu trí tuệ",
        ),
    },
    {
        "anchor": "capital_policies",
        "patterns": (
            "luật thủ đô",
            "thủ đô",
            "chính sách đặc thù",
        ),
        "governing_hints": (
            "luật thủ đô",
        ),
    },
)
COMPONENT_PATTERNS = {
    "hồ sơ": ("hồ sơ", "đơn đề nghị", "tài liệu"),
    "cơ quan": ("cơ quan", "ủy ban", "bộ", "sở", "cục", "nơi nộp", "nộp ở đâu"),
    "thời hạn": ("thời hạn", "bao lâu", "trong bao nhiêu ngày", "khi nào"),
    "trình tự": ("trình tự", "thủ tục", "quy trình", "các bước"),
    "mức phạt": ("mức phạt", "phạt tiền", "xử phạt", "phạt bao nhiêu"),
    "biện pháp khắc phục": ("khắc phục hậu quả", "biện pháp khắc phục"),
    "điều kiện": ("điều kiện", "tiêu chí", "trường hợp", "được hưởng khi"),
    "chính sách hỗ trợ": ("hỗ trợ", "chính sách", "ưu đãi"),
    "thẩm quyền": ("thẩm quyền", "ai có quyền", "cơ quan nào quyết định"),
}
NORM_ROLE_PATTERNS = {
    "procedure": ("hồ sơ", "thủ tục", "trình tự", "quy trình", "thời hạn"),
    "authority": ("thẩm quyền", "trách nhiệm", "cơ quan", "ủy ban", "bộ", "chính phủ"),
    "penalty": ("xử phạt", "mức phạt", "phạt tiền"),
    "remedy": ("khắc phục hậu quả", "biện pháp khắc phục"),
    "condition": ("điều kiện", "tiêu chí", "trường hợp"),
    "support_policy": ("hỗ trợ", "ưu đãi", "miễn", "giảm"),
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
    requested_components = infer_requested_components(normalized, question_type=question_type, answer_shape=answer_shape)
    target_norm_roles = infer_target_norm_roles(normalized, question_type=question_type, answer_shape=answer_shape, intent=intent)
    domain_anchors = infer_domain_anchors(normalized)
    governing_doc_hints = infer_governing_doc_hints(normalized, domain_anchors=domain_anchors)
    subjects, actions, objects, time_or_amount = infer_legal_frame(normalized)
    return PredictedQuestionMetadata(
        intent=intent,
        question_type=question_type,
        answer_shape=answer_shape,
        needs_guidance_docs=guidance,
        target_doc_ids=target_doc_ids or extract_doc_ids(normalized),
        target_article_labels=target_article_labels or extract_article_labels(normalized),
        legal_facets=facets,
        retrieval_bias=retrieval_bias,
        requested_components=requested_components,
        target_norm_roles=target_norm_roles,
        governing_doc_hints=governing_doc_hints,
        domain_anchors=domain_anchors,
        legal_subjects=subjects,
        legal_actions=actions,
        legal_objects=objects,
        time_or_amount=time_or_amount,
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


def infer_must_include_terms(question: str) -> list[str]:
    lowered = question.lower()
    output: list[str] = []
    for _label, variants in PROTECTED_TERM_GROUPS:
        if any(variant in lowered for variant in variants):
            for variant in variants:
                if variant not in output:
                    output.append(variant)
    return output[:8]


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


def infer_domain_anchors(question: str) -> list[str]:
    lowered = question.lower()
    anchors: list[str] = []
    for rule in DOMAIN_ANCHOR_RULES:
        if any(pattern in lowered for pattern in rule["patterns"]) and rule["anchor"] not in anchors:
            anchors.append(rule["anchor"])
    return anchors[:4]


def infer_governing_doc_hints(question: str, *, domain_anchors: list[str] | None = None) -> list[str]:
    lowered = question.lower()
    hints: list[str] = []
    for rule in DOMAIN_ANCHOR_RULES:
        if rule["anchor"] in (domain_anchors or []) or any(pattern in lowered for pattern in rule["patterns"]):
            for hint in rule["governing_hints"]:
                if hint not in hints:
                    hints.append(hint)
    for doc_type in infer_doc_type_hints(question):
        if doc_type not in hints:
            hints.append(doc_type)
    return hints[:6]


def infer_requested_components(question: str, *, question_type: str, answer_shape: str) -> list[str]:
    lowered = question.lower()
    output: list[str] = []
    if question_type == "procedure" or answer_shape == "procedure_steps":
        output.extend(["hồ sơ", "cơ quan", "thời hạn", "trình tự"])
    elif question_type == "penalty":
        output.append("mức phạt")
        if "khắc phục hậu quả" in lowered or "khắc phục" in lowered or answer_shape == "penalty_and_remedy":
            output.append("biện pháp khắc phục")
    elif question_type == "condition":
        output.append("điều kiện")
    elif question_type == "authority":
        output.append("thẩm quyền")

    for label, patterns in COMPONENT_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns) and label not in output:
            output.append(label)
    return output[:6]


def infer_target_norm_roles(question: str, *, question_type: str, answer_shape: str, intent: str) -> list[str]:
    lowered = question.lower()
    roles: list[str] = []
    if question_type == "procedure" or answer_shape == "procedure_steps":
        roles.append("procedure")
    if question_type == "authority" or intent in {"authority", "responsibility"}:
        roles.append("authority")
    if question_type == "penalty" or intent == "penalty":
        roles.append("penalty")
        if "khắc phục hậu quả" in lowered or answer_shape == "penalty_and_remedy":
            roles.append("remedy")
    if question_type == "condition":
        roles.append("condition")
    if intent in {"support_policy", "tax_land"} or question_type == "list":
        roles.append("support_policy")
    for role, patterns in NORM_ROLE_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns) and role not in roles:
            roles.append(role)
    return roles[:5]


def infer_legal_frame(question: str) -> tuple[list[str], list[str], list[str], list[str]]:
    lowered = question.lower()
    subject_patterns = {
        "doanh nghiệp nhỏ và vừa": ("doanh nghiệp nhỏ và vừa",),
        "doanh nghiệp": ("doanh nghiệp", "công ty"),
        "người lao động": ("người lao động", "nhân viên"),
        "người sử dụng lao động": ("người sử dụng lao động",),
        "cơ sở ươm tạo": ("cơ sở ươm tạo",),
        "khu làm việc chung": ("khu làm việc chung",),
    }
    action_patterns = {
        "hưởng hỗ trợ": ("hưởng", "được hỗ trợ", "hỗ trợ"),
        "giữ bản chính": ("giữ bản chính", "giữ bằng cấp", "giữ văn bằng", "giữ chứng chỉ"),
        "xử phạt": ("xử phạt", "phạt", "bị xử lý"),
        "đăng ký": ("đăng ký",),
        "nộp hồ sơ": ("nộp hồ sơ", "đề nghị",),
    }
    object_patterns = {
        "bằng cấp": ("bằng cấp", "văn bằng", "chứng chỉ"),
        "đất đai": ("đất đai", "đất"),
        "thuế": ("thuế", "mã số thuế"),
        "hóa đơn": ("hóa đơn",),
        "nhãn hiệu": ("nhãn hiệu",),
        "sáng chế": ("sáng chế",),
    }
    time_patterns = {
        "thời hạn": ("thời hạn", "bao lâu", "trong bao nhiêu ngày", "ngày"),
        "mức tiền": ("mức phạt", "phạt tiền", "bao nhiêu tiền"),
    }
    return (
        _match_labels(lowered, subject_patterns),
        _match_labels(lowered, action_patterns),
        _match_labels(lowered, object_patterns),
        _match_labels(lowered, time_patterns),
    )


def _match_labels(lowered: str, patterns: dict[str, tuple[str, ...]]) -> list[str]:
    output: list[str] = []
    for label, values in patterns.items():
        if any(value in lowered for value in values) and label not in output:
            output.append(label)
    return output[:6]


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
