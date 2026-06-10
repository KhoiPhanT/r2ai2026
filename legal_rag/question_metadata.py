from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from legal_rag.domain.components import detect_component_requirements, norm_roles_for_components
from legal_rag.schemas.models import GoldQuestionMetadata, PredictedQuestionMetadata, Question
from legal_rag.schemas.models import MetadataSignal, RequestedComponent
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
DEADLINE_MARKERS = ("thời hạn", "bao lâu", "trong bao nhiêu ngày", "thời gian tối đa", "tối đa là bao lâu")
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
    ("doanh nghiệp nhỏ và vừa", ("doanh nghiệp nhỏ và vừa", "doanh nghiệp vừa và nhỏ", "công ty nhỏ và vừa", "sme", "dnnvv")),
    ("cơ sở ươm tạo", ("cơ sở ươm tạo", "ươm tạo")),
    ("khu làm việc chung", ("khu làm việc chung",)),
    ("hỗ trợ", ("hỗ trợ",)),
    ("ưu đãi", ("ưu đãi", "ưu đãi đầu tư")),
    ("thu hút đầu tư", ("thu hút đầu tư", "đầu tư", "phát triển doanh nghiệp")),
    ("khoa học công nghệ", ("khoa học", "công nghệ", "đổi mới sáng tạo", "chuyển đổi số")),
    ("nhân lực", ("nhân lực", "lao động", "nguồn nhân lực")),
    ("thuế", ("thuế", "mã số thuế", "kê khai thuế")),
    ("hóa đơn", ("hóa đơn",)),
    ("đất đai", ("đất đai", "đất", "thuê mặt bằng", "mặt bằng sản xuất", "giá thuê mặt bằng", "chi phí thuê mặt bằng")),
    ("đấu thầu", ("đấu thầu", "nhà thầu")),
    ("bảo hiểm xã hội", ("bảo hiểm xã hội", "bhxh")),
    ("bảo hiểm thất nghiệp", ("bảo hiểm thất nghiệp",)),
    ("hợp đồng lao động", ("hợp đồng lao động",)),
    ("giữ bản chính", ("giữ bản chính", "giữ giấy tờ tùy thân", "giữ văn bằng", "giữ chứng chỉ")),
    ("bằng cấp", ("bằng cấp", "văn bằng", "chứng chỉ")),
    ("sở hữu trí tuệ", ("sở hữu trí tuệ", "nhãn hiệu", "sáng chế", "kiểu dáng", "tác giả")),
    ("xử phạt", ("xử phạt", "mức phạt", "vi phạm")),
    ("khắc phục hậu quả", ("khắc phục hậu quả",)),
    ("thủ tục hồ sơ", ("thủ tục", "hồ sơ", "trình tự", "biểu mẫu")),
    ("thẩm quyền", ("thẩm quyền", "cơ quan", "ủy ban", "bộ", "chính phủ")),
]
PROTECTED_TERM_GROUPS = (
    ("doanh nghiệp nhỏ và vừa", ("doanh nghiệp nhỏ và vừa", "doanh nghiệp vừa và nhỏ", "công ty nhỏ và vừa", "sme", "dnnvv")),
    ("cơ sở ươm tạo", ("cơ sở ươm tạo", "ươm tạo")),
    ("khu làm việc chung", ("khu làm việc chung",)),
    ("khởi nghiệp sáng tạo", ("khởi nghiệp sáng tạo",)),
    ("chuỗi giá trị", ("chuỗi giá trị",)),
    ("quỹ phát triển doanh nghiệp nhỏ và vừa", ("quỹ phát triển doanh nghiệp nhỏ và vừa",)),
    ("thuế", ("thuế", "ưu đãi thuế", "miễn thuế")),
    ("đất đai", ("đất đai", "đất", "thuê mặt bằng", "mặt bằng sản xuất", "giá thuê mặt bằng", "chi phí thuê mặt bằng")),
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
            "doanh nghiệp vừa và nhỏ",
            "công ty nhỏ và vừa",
            "doanh nghiệp siêu nhỏ",
            "cơ sở ươm tạo",
            "khu làm việc chung",
            "khởi nghiệp sáng tạo",
            "chuỗi giá trị",
        ),
        "governing_hints": (
            "80/2021/NĐ-CP",
            "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
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
            "12/2022/NĐ-CP",
            "Nghị định 12/2022/NĐ-CP xử phạt lao động",
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
    "cơ quan": ("cơ quan", "ủy ban", "thẩm quyền", "nơi nộp", "nộp ở đâu"),
    "thời hạn": ("thời hạn", "bao lâu", "trong bao nhiêu ngày", "thời gian tối đa", "tối đa là bao lâu"),
    "trình tự": ("trình tự", "thủ tục", "quy trình", "các bước"),
    "mức phạt": ("mức phạt", "phạt tiền", "phạt bao nhiêu"),
    "biện pháp khắc phục": ("khắc phục hậu quả", "biện pháp khắc phục"),
    "điều kiện": ("điều kiện", "tiêu chí", "trường hợp", "được hưởng khi"),
    "chính sách hỗ trợ": ("hỗ trợ", "chính sách", "ưu đãi"),
    "thuế": ("thuế", "ưu đãi thuế", "miễn thuế"),
    "hóa đơn": ("hóa đơn", "hoá đơn"),
    "đất đai": ("đất đai", "đất", "thuê mặt bằng", "mặt bằng sản xuất", "giá thuê mặt bằng", "chi phí thuê mặt bằng"),
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
    component_requirements = infer_component_requirements(
        normalized,
        question_type=question_type,
        answer_shape=answer_shape,
        intent=intent,
    )
    requested_components = [item.name for item in component_requirements if item.requirement == "required"]
    target_norm_roles = infer_target_norm_roles(
        normalized,
        question_type=question_type,
        answer_shape=answer_shape,
        intent=intent,
        component_requirements=component_requirements,
    )
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
        signals=_question_signals(normalized, target_doc_ids or extract_doc_ids(normalized), target_article_labels or extract_article_labels(normalized), component_requirements),
        component_requirements=component_requirements,
        confidence=max(0.0, min(1.0, float(confidence))),
    )


def infer_question_type(question: str, *, intent: str = "general") -> str:
    lowered = question.lower()
    if any(marker in lowered for marker in AUTHORITY_MARKERS) or intent == "authority":
        return "authority"
    if _is_penalty_question(lowered):
        return "penalty"
    # Comparison questions often contain procedure/condition terms for each side.
    # The comparison frame must win so the planner can allocate reasoning and
    # retrieval budget across both legal rules.
    if any(marker in lowered for marker in COMPARISON_MARKERS) or intent == "comparison":
        return "comparison"
    if any(marker in lowered for marker in PROCEDURE_MARKERS):
        return "procedure"
    if any(marker in lowered for marker in DEADLINE_MARKERS):
        return "deadline"
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
        if any(marker in lowered for marker in ("khắc phục hậu quả", "biện pháp khắc phục", "khắc phục ra sao", "khắc phục thế nào")):
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
    facets: list[str] = []
    seen: set[str] = set()
    for term in legal_terms or []:
        normalized = term.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            facets.append(normalized)
    for component in detect_component_requirements(question):
        key = component.name.lower()
        if key not in seen:
            seen.add(key)
            facets.append(component.name)
    for title in _explicit_document_titles(question):
        key = title.lower()
        if key not in seen:
            seen.add(key)
            facets.append(title)
    return facets[:8]


def infer_must_include_terms(question: str) -> list[str]:
    output = [component.name for component in detect_component_requirements(question)]
    output.extend(_explicit_document_titles(question))
    return list(dict.fromkeys(output))[:8]


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
    # Domain hypotheses come from the corpus lexicon or the LLM. Before retrieval,
    # only an explicitly named legal instrument is safe enough to anchor a regime.
    return _explicit_document_titles(question)[:4]


def infer_governing_doc_hints(question: str, *, domain_anchors: list[str] | None = None) -> list[str]:
    hints = [*extract_doc_ids(question), *_explicit_document_titles(question), *(domain_anchors or [])]
    return list(dict.fromkeys(hints))[:6]


def _explicit_document_titles(question: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:Bộ luật|Luật|Nghị định|Thông tư(?: liên tịch)?|Nghị quyết|Pháp lệnh)\s+"
        r"(?!số\b)([^?;,.]{2,80})",
        re.IGNORECASE,
    )
    output: list[str] = []
    for match in pattern.finditer(question):
        value = " ".join(match.group(0).split()).strip()
        value = re.sub(r"\s+(?:quy định|nêu|cho biết|thì)\b.*$", "", value, flags=re.IGNORECASE).strip()
        if value and value.lower() not in {item.lower() for item in output}:
            output.append(value)
    return output


def infer_component_requirements(
    question: str,
    *,
    question_type: str,
    answer_shape: str,
    intent: str = "general",
) -> list[RequestedComponent]:
    explicit = detect_component_requirements(question)
    by_name = {item.name: item for item in explicit}
    inferred: list[RequestedComponent] = []

    def add_inferred(name: str, confidence: float = 0.75) -> None:
        if name in by_name:
            return
        inferred.append(
            RequestedComponent(
                name=name,
                requirement="optional",
                source_span="",
                confidence=confidence,
                source="question_llm_hint",
            )
        )

    if question_type == "procedure" or answer_shape == "procedure_steps":
        add_inferred("trình tự")
    if question_type == "penalty" or intent == "penalty":
        add_inferred("mức phạt")
    if question_type == "condition" or intent == "condition":
        add_inferred("điều kiện")
    if question_type == "authority" or intent == "authority":
        add_inferred("thẩm quyền")
    return [*explicit, *inferred][:8]


def infer_requested_components(question: str, *, question_type: str, answer_shape: str) -> list[str]:
    return [
        item.name
        for item in infer_component_requirements(
            question,
            question_type=question_type,
            answer_shape=answer_shape,
        )
        if item.requirement == "required"
    ]


def infer_target_norm_roles(
    question: str,
    *,
    question_type: str,
    answer_shape: str,
    intent: str,
    component_requirements: list[RequestedComponent] | None = None,
) -> list[str]:
    lowered = question.lower()
    requirements = component_requirements or infer_component_requirements(
        question,
        question_type=question_type,
        answer_shape=answer_shape,
        intent=intent,
    )
    roles = norm_roles_for_components(requirements, required_only=True)
    if intent == "responsibility" and "responsibility" not in roles:
        roles.append("responsibility")
    if intent == "support_policy" and any(
        item.name == "chính sách hỗ trợ" and item.requirement == "required" for item in requirements
    ):
        roles.append("support_policy")
    required_names = {item.name for item in requirements if item.requirement == "required"}
    if question_type == "penalty" and "mức phạt" in required_names and "penalty" not in roles:
        roles.append("penalty")
    if answer_shape == "penalty_and_remedy" and "biện pháp khắc phục" in lowered and "remedy" not in roles:
        roles.append("remedy")
    return roles[:5]


def _question_signals(
    question: str,
    doc_ids: list[str],
    article_labels: list[str],
    components: list[RequestedComponent],
) -> list[MetadataSignal]:
    signals: list[MetadataSignal] = []
    for doc_id in doc_ids:
        signals.append(MetadataSignal("doc_id", doc_id, "question_rule", doc_id, 1.0, "hard"))
    for label in article_labels:
        signals.append(MetadataSignal("article_label", label, "question_rule", label, 1.0, "hard"))
    for component in components:
        signals.append(
            MetadataSignal(
                "component",
                component.name,
                component.source,
                component.source_span,
                component.confidence,
                "required" if component.requirement == "required" else "advisory",
            )
        )
    for value in _numeric_signals(question):
        signals.append(MetadataSignal("time_or_amount", value, "question_rule", value, 1.0, "required"))
    return signals


def _numeric_signals(question: str) -> list[str]:
    pattern = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|đồng|triệu|tỷ|ngày|tháng|năm)\b", re.IGNORECASE)
    return [match.group(0) for match in pattern.finditer(question)]


def _is_penalty_question(lowered: str) -> bool:
    if any(marker in lowered for marker in ("mức phạt", "xử phạt", "bị phạt", "phạt bao nhiêu", "khắc phục hậu quả")):
        return True
    if any(marker in lowered for marker in ("bị xử lý thế nào", "bị xử lý như thế nào")):
        return True
    if "bị xử lý" in lowered and any(marker in lowered for marker in ("vi phạm", "không ", "trái quy định")):
        return True
    return False


def infer_legal_frame(question: str) -> tuple[list[str], list[str], list[str], list[str]]:
    # Before retrieval, the deterministic layer owns only exact structural facts.
    # Semantic subjects/actions/objects come from the schema-constrained planner
    # and corpus-derived lexicon, where every expansion has provenance.
    return [], [], [], _numeric_signals(question)


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
