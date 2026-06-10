from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from legal_rag.domain.components import COMPONENT_REGISTRY, component_matches, infer_evidence_components
from legal_rag.question_metadata import infer_legal_facets, infer_must_include_terms
from legal_rag.schemas.models import ArticleNode, EvidenceMetadata
from legal_rag.utils.text import extract_article_labels

FACET_STOPWORDS = {"và", "các", "những", "theo", "của", "cho", "khi", "với"}


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    issues: list[str]
    hard_issues: list[str] | None = None
    repairable_issues: list[str] | None = None
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        self.hard_issues = list(self.hard_issues or [])
        self.repairable_issues = list(self.repairable_issues or [])
        self.warnings = list(self.warnings or [])

    @property
    def needs_repair(self) -> bool:
        return bool(self.repairable_issues)


def verify_prediction_evidence(answer: str, articles: list[ArticleNode]) -> VerificationResult:
    issues: list[str] = []
    cited = {label.lower() for label in extract_article_labels(answer)}
    available = {article.article_label.lower() for article in articles}

    if articles and not cited:
        issues.append("answer_missing_article_citation")
    for label in sorted(cited):
        if label not in available:
            issues.append(f"citation_not_in_retrieved_set:{label}")

    article_keys = [article.article_key for article in articles]
    if len(article_keys) != len(set(article_keys)):
        issues.append("duplicate_article_key")

    return VerificationResult(ok=not issues, issues=issues)


def verify_used_evidence_answer(
    answer: str,
    used_articles: list[ArticleNode],
    candidate_articles: Iterable[ArticleNode] | None = None,
    question: str = "",
    *,
    required_components: list[str] | None = None,
    target_norm_roles: list[str] | None = None,
    covered_components: list[str] | None = None,
    insufficient_components: list[str] | None = None,
    claims: list[dict[str, Any]] | None = None,
    evidence_by_id: dict[str, ArticleNode] | None = None,
) -> VerificationResult:
    hard: list[str] = []
    repairable: list[str] = []
    warnings: list[str] = []
    if not used_articles:
        hard.append("answer_missing_used_evidence")
        return _verification_result(hard, repairable, warnings)

    cited = [label.lower() for label in extract_article_labels(answer)]
    if not cited:
        repairable.append("answer_missing_article_citation")
    used_by_label: dict[str, list[ArticleNode]] = {}
    for article in used_articles:
        used_by_label.setdefault(article.article_label.lower(), []).append(article)

    candidate_keys = {article.article_key for article in candidate_articles or []}
    for article in used_articles:
        if candidate_keys and article.article_key not in candidate_keys:
            hard.append(f"used_evidence_not_retrieved:{article.article_key}")

    for label in sorted(set(cited)):
        matches = used_by_label.get(label, [])
        if not matches:
            hard.append(f"citation_not_in_used_evidence:{label}")
        elif len(matches) > 1 and not _answer_disambiguates(answer, matches):
            hard.append(f"ambiguous_article_label:{label}")
    cited_set = set(cited)
    for article in used_articles:
        if article.article_label.lower() not in cited_set:
            repairable.append(f"used_evidence_not_cited:{article.article_key}")

    article_keys = [article.article_key for article in used_articles]
    if len(article_keys) != len(set(article_keys)):
        hard.append("duplicate_article_key")

    semantic_issues = _semantic_sanity_issues(question, used_articles, insufficient_components or [], answer)
    warnings.extend(semantic_issues)
    repairable.extend(_authority_directness_issues(question, used_articles, target_norm_roles or [], answer))
    inferred_covered_components = _infer_covered_components(answer, used_articles, covered_components or [])
    component_issues, role_warnings = _component_coverage_issues(
            used_articles,
            required_components or [],
            target_norm_roles or [],
            inferred_covered_components,
            insufficient_components or [],
            answer,
        )
    repairable.extend(component_issues)
    warnings.extend(role_warnings)
    claim_hard, claim_repairable = _claim_support_issues(claims or [], evidence_by_id or {})
    hard.extend(claim_hard)
    repairable.extend(claim_repairable)

    return _verification_result(hard, repairable, warnings)


def build_evidence_metadata(articles: list[ArticleNode]) -> list[EvidenceMetadata]:
    output: list[EvidenceMetadata] = []
    for article in articles:
        support_text = str(article.metadata.get("support_snippet") or article.text[:1600])
        components = infer_evidence_components(support_text)
        roles = [str(item) for item in article.metadata.get("norm_roles", []) if str(item).strip()]
        support_type = str(article.metadata.get("evidence_admission", {}).get("support_type") or "")
        if not support_type:
            support_type = "cross_reference" if _is_cross_reference_only_text(support_text.lower()) else "direct"
        output.append(
            EvidenceMetadata(
                article_key=article.article_key,
                supported_components=components,
                norm_roles=roles,
                support_type=support_type,
            )
        )
    return output


def _verification_result(hard: list[str], repairable: list[str], warnings: list[str]) -> VerificationResult:
    issues = [*hard, *repairable]
    return VerificationResult(
        ok=not hard,
        issues=issues,
        hard_issues=hard,
        repairable_issues=repairable,
        warnings=warnings,
    )


def _answer_disambiguates(answer: str, articles: list[ArticleNode]) -> bool:
    lowered = answer.lower()
    return all(article.doc_id.lower() in lowered or article.title_for_submission.lower() in lowered for article in articles)


def _semantic_sanity_issues(
    question: str,
    used_articles: list[ArticleNode],
    insufficient_components: list[str] | None = None,
    answer: str = "",
) -> list[str]:
    if not question or not used_articles:
        return []
    combined = " ".join(
        f"{article.title_for_submission} {article.article_title} {article.metadata.get('support_snippet', '')} {article.text[:1200]}".lower()
        for article in used_articles
    )
    anchors = infer_must_include_terms(question)
    insufficient_aliases = _insufficient_anchor_aliases(insufficient_components or [], answer)
    if insufficient_aliases:
        anchors = [anchor for anchor in anchors if anchor.lower() not in insufficient_aliases]
    if anchors:
        matched = [anchor for anchor in anchors if _anchor_matches(anchor, combined)]
        if not matched:
            return [f"semantic_domain_mismatch:{'|'.join(anchors[:3])}"]
    facets = [facet for facet in infer_legal_facets(question) if len(facet) > 3]
    if facets:
        matched_facets = [
            facet
            for facet in facets
            if facet.lower() in combined
            or any(token in combined for token in facet.lower().split() if len(token) > 1 and token not in FACET_STOPWORDS)
        ]
        if not matched_facets:
            return [f"semantic_facet_mismatch:{'|'.join(facets[:3])}"]
    return []


def _authority_directness_issues(
    question: str,
    used_articles: list[ArticleNode],
    target_norm_roles: list[str],
    answer: str,
) -> list[str]:
    if not used_articles:
        return []
    lowered_question = question.lower()
    needs_authority = "authority" in {role.lower() for role in target_norm_roles} or any(
        marker in lowered_question for marker in ("thẩm quyền", "cơ quan nào", "ai có quyền")
    )
    if not needs_authority:
        return []
    combined = " ".join(
        f"{article.title_for_submission} {article.article_title} {article.metadata.get('support_snippet', '')} {article.text[:900]}".lower()
        for article in used_articles
    )
    if _is_cross_reference_only_authority_text(combined) and not _has_direct_authority_signal(combined):
        return ["authority_evidence_is_cross_reference_only"]
    answer_lower = answer.lower()
    if any(term in answer_lower for term in ("cơ quan quản lý thuế", "cơ quan thuế")) and not any(
        term in combined for term in ("cơ quan quản lý thuế", "cơ quan thuế có thẩm quyền", "cơ quan thuế")
    ):
        return ["authority_answer_not_supported_by_evidence"]
    return []


def _is_cross_reference_only_authority_text(text: str) -> bool:
    pointer_markers = (
        "được thực hiện theo quy định của pháp luật",
        "thực hiện theo quy định của pháp luật",
        "áp dụng quy định của pháp luật",
        "theo quy định của pháp luật về xử phạt",
    )
    if not any(marker in text for marker in pointer_markers):
        return False
    return "thẩm quyền" in text or "xử phạt" in text


def _is_cross_reference_only_text(text: str) -> bool:
    pointer_markers = (
        "được thực hiện theo quy định",
        "thực hiện theo quy định",
        "theo quy định của pháp luật",
        "theo quy định tại điều",
        "đáp ứng quy định tại điều",
        "quy định chi tiết tại",
    )
    if not any(marker in text for marker in pointer_markers):
        return False
    substantive_markers = (
        "bao gồm",
        "phạt tiền từ",
        "thời hạn",
        "có thẩm quyền",
        "điều kiện",
        "được hưởng",
        "phải thông báo",
        "hồ sơ gồm",
    )
    return not any(marker in text for marker in substantive_markers)


def _claim_support_issues(
    claims: list[dict[str, Any]],
    evidence_by_id: dict[str, ArticleNode],
) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    repairable: list[str] = []
    stopwords = FACET_STOPWORDS | {"được", "phải", "trong", "một", "này", "đó", "theo", "quy định"}
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            hard.append(f"claim_invalid:{index}")
            continue
        text = " ".join(str(claim.get("claim") or "").split()).strip()
        ids = [str(item).strip() for item in claim.get("evidence_ids", []) if str(item).strip()]
        if not text or not ids:
            hard.append(f"claim_missing_support:{index}")
            continue
        unknown = [item for item in ids if item not in evidence_by_id]
        if unknown:
            hard.append(f"claim_unknown_evidence:{index}:{'|'.join(unknown)}")
            continue
        articles = [evidence_by_id[item] for item in ids]
        evidence_text = " ".join(
            f"{article.doc_id} {article.article_label} {article.article_title} "
            f"{article.metadata.get('support_snippet', '')} {article.text[:1800]}"
            for article in articles
        ).lower()
        for value in _specific_claim_values(text):
            if _normalize_claim_value(value) not in _normalize_claim_value(evidence_text):
                hard.append(f"claim_value_not_supported:{index}:{value}")
        claim_doc_ids = re.findall(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+(?:\d+)?\b", text.upper())
        if claim_doc_ids and not all(doc_id.lower() in evidence_text for doc_id in claim_doc_ids):
            hard.append(f"claim_doc_not_supported:{index}")
        claim_labels = {label.lower() for label in extract_article_labels(text)}
        evidence_labels = {article.article_label.lower() for article in articles}
        if claim_labels and not claim_labels.issubset(evidence_labels):
            hard.append(f"claim_article_not_supported:{index}")
        claim_tokens = {
            token
            for token in re.findall(r"[\wÀ-ỹ]+", text.lower())
            if len(token) >= 4 and token not in stopwords
        }
        evidence_tokens = set(re.findall(r"[\wÀ-ỹ]+", evidence_text))
        if len(claim_tokens) >= 4 and not (claim_tokens & evidence_tokens):
            repairable.append(f"claim_low_lexical_support:{index}")
    return list(dict.fromkeys(hard)), list(dict.fromkeys(repairable))


def _specific_claim_values(text: str) -> list[str]:
    patterns = (
        r"\b\d+(?:[.,]\d+)*\s*%",
        r"\b\d+(?:[.,]\d+)*\s*(?:đồng|triệu|tỷ)\b",
        r"\b\d+\s*(?:ngày|tháng|năm|giờ)\b",
    )
    output: list[str] = []
    for pattern in patterns:
        output.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return output


def _normalize_claim_value(value: str) -> str:
    return re.sub(r"[\s.,]", "", value.lower())


def _has_direct_authority_signal(text: str) -> bool:
    direct_patterns = (
        "có thẩm quyền",
        "thẩm quyền của",
        "chủ tịch ủy ban",
        "chủ tịch ubnd",
        "chánh thanh tra",
        "thanh tra viên",
        "cục trưởng",
        "chi cục trưởng",
        "tổng cục trưởng",
        "trưởng đoàn thanh tra",
        "cơ quan thuế có thẩm quyền",
        "cơ quan quản lý thuế có thẩm quyền",
    )
    return any(pattern in text for pattern in direct_patterns)


def _anchor_matches(anchor: str, combined: str) -> bool:
    anchor_lower = anchor.lower()
    if anchor_lower in combined:
        return True
    if anchor_lower in {name.lower() for name in COMPONENT_REGISTRY}:
        return component_matches(anchor_lower, combined, evidence=True)
    anchor_aliases = {
        "đất": ("mặt bằng", "thuê mặt bằng", "thuê đất", "tiền thuê đất", "tiền sử dụng đất"),
        "đất đai": ("mặt bằng", "thuê mặt bằng", "thuê đất", "tiền thuê đất", "tiền sử dụng đất"),
        "thuế": ("ưu đãi thuế", "miễn thuế", "giảm thuế", "thuế sử dụng đất", "thu nhập doanh nghiệp"),
        "hóa đơn": ("hoá đơn",),
        "hoá đơn": ("hóa đơn",),
    }
    return any(alias in combined for alias in anchor_aliases.get(anchor_lower, ()))


def _insufficient_anchor_aliases(insufficient_components: list[str], answer: str) -> set[str]:
    aliases_by_component = {
        "thuế": ("thuế", "ưu đãi thuế", "miễn thuế", "giảm thuế", "thu nhập doanh nghiệp"),
        "hóa đơn": ("hóa đơn", "hoá đơn"),
        "đất đai": ("đất đai", "đất", "thuê đất", "tiền thuê đất", "mặt bằng", "thuê mặt bằng"),
    }
    aliases: set[str] = set()
    for component in insufficient_components:
        component_lower = component.lower()
        if not _answer_acknowledges_insufficient_component(answer, component_lower):
            continue
        aliases.update(alias.lower() for alias in aliases_by_component.get(component_lower, (component_lower,)))
    return aliases


def _component_coverage_issues(
    used_articles: list[ArticleNode],
    required_components: list[str],
    target_norm_roles: list[str],
    covered_components: list[str],
    insufficient_components: list[str],
    answer: str,
) -> tuple[list[str], list[str]]:
    if not used_articles:
        return [], []
    combined = " ".join(
        f"{article.metadata.get('support_snippet', '')} {article.article_title} {article.text[:1200]}".lower()
        for article in used_articles
    )
    norm_roles = {str(role).lower() for article in used_articles for role in article.metadata.get("norm_roles", [])}
    covered = {item.lower() for item in covered_components}
    insufficient = {item.lower() for item in insufficient_components}
    issues: list[str] = []
    warnings: list[str] = []
    for component in required_components:
        if component.lower() not in covered:
            if component.lower() in insufficient and _answer_acknowledges_insufficient_component(answer, component):
                continue
            issues.append(f"component_coverage_missing:{component}")
    role_aliases = {
        "procedure": ("procedure", "hồ sơ", "thủ tục", "trình tự", "thời hạn"),
        "authority": ("authority", "thẩm quyền"),
        "penalty": ("penalty", "mức phạt", "xử phạt"),
        "remedy": ("remedy", "biện pháp khắc phục"),
        "condition": ("condition", "điều kiện"),
        "support_policy": ("support_policy", "hỗ trợ", "ưu đãi"),
        "responsibility": ("responsibility", "trách nhiệm", "nghĩa vụ"),
    }
    for role in target_norm_roles:
        aliases = role_aliases.get(role.lower(), ())
        if aliases and not any(alias in norm_roles or alias in combined for alias in aliases):
            warnings.append(f"norm_role_mismatch:{role}")
    return issues, warnings


def _answer_acknowledges_insufficient_component(answer: str, component: str) -> bool:
    lowered = answer.lower()
    component_lower = component.lower()
    if component_lower not in lowered:
        return False
    insufficiency_markers = (
        "chưa đủ căn cứ",
        "không đủ căn cứ",
        "chưa tìm thấy căn cứ",
        "không tìm thấy căn cứ",
        "evidence chưa đủ",
        "căn cứ được cung cấp chưa",
    )
    return any(marker in lowered for marker in insufficiency_markers)


def _infer_covered_components(
    answer: str,
    used_articles: list[ArticleNode],
    covered_components: list[str],
) -> list[str]:
    answer_text = answer.lower()
    evidence_text = " ".join(
        f"{article.article_title} {article.metadata.get('support_snippet') or article.text[:2000]}".lower()
        for article in used_articles
    )
    candidates = set(covered_components) | set(infer_evidence_components(evidence_text))
    inferred: set[str] = set()
    for label in candidates:
        if component_matches(label, answer_text) and component_matches(label, evidence_text):
            inferred.add(label)
    return sorted(inferred)
