from __future__ import annotations

import re
from dataclasses import dataclass

from legal_rag.schemas.models import RequestedComponent


@dataclass(frozen=True, slots=True)
class ComponentDefinition:
    name: str
    question_patterns: tuple[str, ...]
    evidence_patterns: tuple[str, ...]
    norm_roles: tuple[str, ...] = ()
    retrieval_weight: float = 1.0


COMPONENT_REGISTRY: dict[str, ComponentDefinition] = {
    "hồ sơ": ComponentDefinition(
        "hồ sơ",
        (r"\bhồ sơ\b", r"giấy tờ (?:gì|nào)", r"tài liệu (?:gì|nào)", r"đơn (?:gì|nào)"),
        (r"\bhồ sơ\b", r"đơn đề nghị", r"tài liệu", r"giấy tờ"),
        ("procedure",),
        1.25,
    ),
    "trình tự": ComponentDefinition(
        "trình tự",
        (r"\btrình tự\b", r"\bthủ tục\b", r"\bquy trình\b", r"các bước (?:gì|nào)"),
        (r"\btrình tự\b", r"\bthủ tục\b", r"\bquy trình\b", r"bước\s+\d+"),
        ("procedure",),
        1.25,
    ),
    "cơ quan": ComponentDefinition(
        "cơ quan",
        (r"cơ quan nào", r"nộp (?:ở|tại) đâu", r"nơi (?:nộp|tiếp nhận)"),
        (r"cơ quan", r"ủy ban", r"\bbộ\b", r"\bsở\b", r"\bcục\b", r"nơi tiếp nhận"),
        ("authority",),
        1.2,
    ),
    "thẩm quyền": ComponentDefinition(
        "thẩm quyền",
        (r"\bthẩm quyền\b", r"ai có quyền", r"ai (?:được|có thể) quyết định", r"cơ quan nào (?:có quyền|quyết định)"),
        (r"\bthẩm quyền\b", r"có quyền", r"có thẩm quyền", r"quyết định"),
        ("authority",),
        1.35,
    ),
    "thời hạn": ComponentDefinition(
        "thời hạn",
        (r"\bbao lâu\b", r"\bthời hạn\b", r"thời gian (?:tối đa|bao lâu)", r"trong bao nhiêu (?:ngày|tháng|năm)"),
        (r"\bthời hạn\b", r"\bthời gian\b", r"\btối đa\b", r"\d+\s*(?:ngày|tháng|năm)"),
        (),
        1.25,
    ),
    "mức phạt": ComponentDefinition(
        "mức phạt",
        (
            r"mức phạt",
            r"bị phạt (?:bao nhiêu|thế nào|như thế nào)",
            r"bị xử lý (?:thế nào|như thế nào)",
            r"phạt bao nhiêu",
            r"xử phạt (?:bao nhiêu|thế nào|như thế nào)",
        ),
        (r"mức phạt", r"phạt tiền", r"phạt\s+(?:từ|đến)\s*[\d.]", r"từ\s*[\d.]+\s*đồng\s*đến\s*[\d.]+\s*đồng"),
        ("penalty",),
        1.4,
    ),
    "biện pháp khắc phục": ComponentDefinition(
        "biện pháp khắc phục",
        (r"khắc phục hậu quả", r"biện pháp khắc phục", r"phải khắc phục"),
        (r"khắc phục hậu quả", r"biện pháp khắc phục"),
        ("remedy",),
        1.35,
    ),
    "điều kiện": ComponentDefinition(
        "điều kiện",
        (
            r"\bđiều kiện\b",
            r"\btiêu chí\b",
            r"đáp ứng (?:những )?yêu cầu",
            r"yêu cầu (?:gì|nào)",
            r"được hưởng khi",
            r"thuộc (?:những )?trường hợp",
            r"trường hợp nào (?:được|thì)",
        ),
        (r"\bđiều kiện\b", r"\btiêu chí\b", r"\byêu cầu\b", r"trường hợp", r"phải đáp ứng", r"bao gồm"),
        ("condition",),
        1.3,
    ),
    "chính sách hỗ trợ": ComponentDefinition(
        "chính sách hỗ trợ",
        (r"chính sách hỗ trợ (?:gì|nào)", r"được hỗ trợ (?:những gì|gì)", r"hỗ trợ nào"),
        (r"\bhỗ trợ\b", r"\bưu đãi\b", r"được miễn", r"được giảm"),
        ("support_policy",),
        1.15,
    ),
    "thuế": ComponentDefinition(
        "thuế",
        (r"\bthuế\b", r"mã số thuế", r"miễn thuế", r"giảm thuế", r"ưu đãi thuế"),
        (r"\bthuế\b", r"mã số thuế", r"miễn thuế", r"giảm thuế", r"ưu đãi thuế"),
        (),
        1.2,
    ),
    "hóa đơn": ComponentDefinition(
        "hóa đơn",
        (r"h[oó]a đơn",),
        (r"h[oó]a đơn",),
        (),
        1.2,
    ),
    "đất đai": ComponentDefinition(
        "đất đai",
        (r"\bđất đai\b", r"(?:chính sách|ưu đãi|hỗ trợ)\s+(?:về\s+)?(?:thuê\s+)?đất", r"về\s+(?:thuế\s+và\s+)?đất đai"),
        (r"\bđất đai\b", r"thuê đất", r"tiền thuê đất", r"thuê mặt bằng", r"mặt bằng"),
        (),
        1.2,
    ),
    "trách nhiệm": ComponentDefinition(
        "trách nhiệm",
        (r"\btrách nhiệm\b", r"có nghĩa vụ gì", r"phải thực hiện những gì"),
        (r"\btrách nhiệm\b", r"\bnghĩa vụ\b", r"có trách nhiệm", r"phải thực hiện"),
        ("responsibility",),
        1.2,
    ),
}


def component_aliases(name: str, *, evidence: bool = True) -> tuple[str, ...]:
    definition = COMPONENT_REGISTRY.get(name.lower())
    if not definition:
        return (re.escape(name.lower()),)
    return definition.evidence_patterns if evidence else definition.question_patterns


def component_matches(name: str, text: str, *, evidence: bool = True) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in component_aliases(name, evidence=evidence))


def detect_component_requirements(question: str) -> list[RequestedComponent]:
    lowered = question.lower()
    output: list[RequestedComponent] = []
    for name, definition in COMPONENT_REGISTRY.items():
        match = _first_match(definition.question_patterns, lowered)
        if match is None:
            continue
        if name == "mức phạt" and _is_non_amount_penalty_context(lowered, match.start(), match.end()):
            continue
        output.append(
            RequestedComponent(
                name=name,
                requirement="required",
                source_span=question[match.start() : match.end()],
                confidence=1.0,
                source="question_rule",
            )
        )
    return _dedupe_requirements(output)


def infer_evidence_components(text: str) -> list[str]:
    return [name for name in COMPONENT_REGISTRY if component_matches(name, text, evidence=True)]


def norm_roles_for_components(components: list[RequestedComponent], *, required_only: bool = True) -> list[str]:
    roles: list[str] = []
    for component in components:
        if required_only and component.requirement != "required":
            continue
        definition = COMPONENT_REGISTRY.get(component.name)
        for role in definition.norm_roles if definition else ():
            if role not in roles:
                roles.append(role)
    return roles


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns]
    present = [match for match in matches if match is not None]
    return min(present, key=lambda match: match.start()) if present else None


def _is_non_amount_penalty_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 30) : min(len(text), end + 45)]
    if any(marker in window for marker in ("bao nhiêu", "mức phạt", "phạt tiền", "xử phạt thế nào", "xử phạt như thế nào", "bị xử lý thế nào", "bị xử lý như thế nào")):
        return False
    return any(marker in window for marker in ("tiền phạt nộp thừa", "nộp thừa tiền thuế và tiền phạt", "hoàn trả tiền phạt"))


def _dedupe_requirements(values: list[RequestedComponent]) -> list[RequestedComponent]:
    output: list[RequestedComponent] = []
    seen: set[str] = set()
    for value in values:
        if value.name in seen:
            continue
        seen.add(value.name)
        output.append(value)
    return output
