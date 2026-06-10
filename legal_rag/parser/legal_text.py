from __future__ import annotations

import re
from typing import Any

from legal_rag.schemas.models import ArticleNode, DocumentRecord
from legal_rag.utils.text import ARTICLE_HEADING_PATTERN, extract_article_labels, extract_doc_ids, normalize_text

CLAUSE_HEADING_PATTERN = re.compile(r"(?m)^\s*(\d+)\.\s+(.*)$")
POINT_HEADING_PATTERN = re.compile(r"(?m)^\s*([a-zđ])\)\s+(.*)$", re.IGNORECASE)
PART_HEADING_PATTERN = re.compile(r"(?m)^\s*(PHẦN\s+[IVXLC0-9]+)\s*$", re.IGNORECASE)
CHAPTER_HEADING_PATTERN = re.compile(r"(?m)^\s*(CHƯƠNG\s+[IVXLC0-9]+)\s*$", re.IGNORECASE)
SECTION_HEADING_PATTERN = re.compile(r"(?m)^\s*(MỤC\s+\d+[A-Z]?)\s*$", re.IGNORECASE)
APPENDIX_HEADING_PATTERN = re.compile(r"(?m)^\s*(PHỤ LỤC(?:\s+[IVXLC0-9]+)?)\s*$", re.IGNORECASE)

GUIDES_MARKERS = ("quy định chi tiết", "hướng dẫn thi hành", "biện pháp để hướng dẫn thi hành")
AMENDS_MARKERS = ("sửa đổi, bổ sung", "sửa đổi", "bổ sung")
REPLACES_MARKERS = ("thay thế", "thay cho")
ABOLISHES_MARKERS = ("bãi bỏ", "bãi nhiệm", "hết hiệu lực", "chấm dứt hiệu lực")


def parse_document(doc: DocumentRecord) -> list[ArticleNode]:
    text = normalize_text(doc.raw_text)
    doc_relations = _document_relations(doc, text)
    headings = list(ARTICLE_HEADING_PATTERN.finditer(text))
    if not headings:
        return [_article_from_span(doc, "Điều 1", "", text, synthetic=True, document_relations=doc_relations)]

    articles: list[ArticleNode] = []
    for index, match in enumerate(headings):
        start = match.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        article_label = _canonical_article_label(match.group(1))
        article_title = normalize_text(match.group(2))
        article_text = normalize_text(text[start:end])
        context_text = text[max(0, start - 2000) : start]
        article_metadata = _article_structure(article_text, article_label, article_title, doc_relations, context_text)
        articles.append(
            _article_from_span(
                doc,
                article_label,
                article_title,
                article_text,
                article_order=index + 1,
                article_metadata=article_metadata,
                document_relations=doc_relations,
            )
        )
    return articles


def _canonical_article_label(raw: str) -> str:
    suffix = raw.split(None, 1)[1].strip()
    return f"Điều {suffix}"


def _article_from_span(
    doc: DocumentRecord,
    article_label: str,
    article_title: str,
    text: str,
    synthetic: bool = False,
    article_order: int = 1,
    article_metadata: dict[str, Any] | None = None,
    document_relations: dict[str, Any] | None = None,
) -> ArticleNode:
    article_key = f"{doc.doc_id}|{doc.title_for_submission}|{article_label}"
    metadata = dict(doc.metadata)
    metadata["synthetic_article"] = synthetic
    metadata.setdefault("chunk_type", "article")
    metadata.setdefault("node_type", "article")
    metadata.setdefault("doc_id", doc.doc_id)
    metadata.setdefault("article_label", article_label)
    metadata.setdefault("article_order", article_order)
    metadata.setdefault("law_title", doc.title_for_submission)
    if document_relations:
        metadata.setdefault("document_relations", document_relations)
        metadata.setdefault("graph_neighbors", _graph_neighbors(document_relations))
    if article_metadata:
        metadata.update(article_metadata)
    return ArticleNode(
        article_key=article_key,
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        title_for_submission=doc.title_for_submission,
        article_label=article_label,
        article_title=article_title,
        text=text,
        status=doc.status,
        source_url=doc.source_url,
        metadata=metadata,
    )


def _document_relations(doc: DocumentRecord, text: str) -> dict[str, Any]:
    header = normalize_text(str(doc.metadata.get("header_text") or ""))
    title = normalize_text(doc.title_for_submission)
    lines = text.splitlines()
    intro = normalize_text("\n".join(lines[:60]))
    closing = normalize_text("\n".join(lines[-100:]))
    relation_scope = "\n".join([title, header, intro, closing])
    sentences = _relation_sentences(relation_scope)
    references_only: list[dict[str, Any]] = []
    guides_doc_ids: list[str] = []
    amends_doc_ids: list[str] = []
    replaces_doc_ids: list[str] = []
    abolishes_doc_ids: list[str] = []
    for sentence in sentences:
        doc_ids = [item for item in extract_doc_ids(sentence) if item.upper() != doc.doc_id.upper()]
        if not doc_ids:
            if "căn cứ" in sentence.lower():
                references_only.extend(_extract_references(sentence, doc.doc_id))
            continue
        lowered = sentence.lower()
        if "căn cứ" in lowered:
            references_only.extend(_extract_references(sentence, doc.doc_id))
            continue
        precise_doc_ids = doc_ids[:3]
        if _contains_any(lowered, GUIDES_MARKERS):
            guides_doc_ids.extend(precise_doc_ids)
        if _contains_any(lowered, AMENDS_MARKERS):
            amends_doc_ids.extend(precise_doc_ids)
        if _contains_any(lowered, REPLACES_MARKERS):
            replaces_doc_ids.extend(precise_doc_ids)
        if _contains_any(lowered, ABOLISHES_MARKERS):
            abolishes_doc_ids.extend(precise_doc_ids)
        if not any(
            _contains_any(lowered, markers)
            for markers in (GUIDES_MARKERS, AMENDS_MARKERS, REPLACES_MARKERS, ABOLISHES_MARKERS)
        ):
            references_only.extend(_extract_references(sentence, doc.doc_id))
    references = _extract_references(text, doc.doc_id)
    return {
        "guides_doc_ids": _dedupe(guides_doc_ids),
        "amends_doc_ids": _dedupe(amends_doc_ids),
        "replaces_doc_ids": _dedupe(replaces_doc_ids),
        "abolishes_doc_ids": _dedupe(abolishes_doc_ids),
        "references_only": _dedupe_dicts(references_only),
        "references": references,
    }


def _relation_sentences(text: str) -> list[str]:
    chunks: list[str] = []
    for block in text.splitlines():
        block = normalize_text(block)
        if not block:
            continue
        pieces = re.split(r"(?<=[.;:])\s+|(?<=\.)\n+", block)
        for piece in pieces:
            piece = normalize_text(piece)
            if piece:
                chunks.append(piece)
    return chunks


def _article_structure(
    article_text: str,
    article_label: str,
    article_title: str,
    doc_relations: dict[str, Any],
    context_text: str,
) -> dict[str, Any]:
    structure_source = f"{context_text}\n{article_text}"
    part_label = _last_match_text(PART_HEADING_PATTERN, structure_source)
    chapter_label = _last_match_text(CHAPTER_HEADING_PATTERN, structure_source)
    section_label = _last_match_text(SECTION_HEADING_PATTERN, structure_source)
    appendix_label = _last_match_text(APPENDIX_HEADING_PATTERN, structure_source)
    clause_nodes = _parse_clauses(article_text, article_label)
    references = _extract_references(article_text, "")
    norm_roles = _infer_norm_roles(article_title, article_text, doc_relations)
    metadata = {
        "part_label": part_label,
        "chapter_label": chapter_label,
        "section_label": section_label,
        "appendix_label": appendix_label,
        "clause_nodes": clause_nodes,
        "references": references or doc_relations.get("references", []),
        "norm_roles": norm_roles,
        "primary_norm_role": norm_roles[0] if norm_roles else "content",
    }
    return metadata


def _parse_clauses(article_text: str, article_label: str) -> list[dict[str, Any]]:
    body_lines = article_text.splitlines()[1:] if article_text.splitlines() else []
    body = "\n".join(body_lines).strip()
    clause_matches = list(CLAUSE_HEADING_PATTERN.finditer(body))
    if not clause_matches:
        return []
    clauses: list[dict[str, Any]] = []
    for index, match in enumerate(clause_matches):
        start = match.start()
        end = clause_matches[index + 1].start() if index + 1 < len(clause_matches) else len(body)
        clause_label = f"Khoản {match.group(1)}"
        clause_text = normalize_text(body[start:end])
        point_nodes = _parse_points(clause_text, article_label, clause_label)
        clauses.append(
            {
                "span_key": f"{article_label}|{clause_label}",
                "level": "clause",
                "label": clause_label,
                "text": clause_text,
                "point_nodes": point_nodes,
            }
        )
    return clauses


def _parse_points(clause_text: str, article_label: str, clause_label: str) -> list[dict[str, Any]]:
    body_lines = clause_text.splitlines()[1:] if clause_text.splitlines() else []
    body = "\n".join(body_lines).strip()
    point_matches = list(POINT_HEADING_PATTERN.finditer(body))
    if not point_matches:
        return []
    points: list[dict[str, Any]] = []
    for index, match in enumerate(point_matches):
        start = match.start()
        end = point_matches[index + 1].start() if index + 1 < len(point_matches) else len(body)
        point_label = f"Điểm {match.group(1).lower()}"
        point_text = normalize_text(body[start:end])
        points.append(
            {
                "span_key": f"{article_label}|{clause_label}|{point_label}",
                "parent_span_key": f"{article_label}|{clause_label}",
                "level": "point",
                "label": point_label,
                "text": point_text,
            }
        )
    return points


def _extract_references(text: str, own_doc_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines:
        doc_ids = [item for item in extract_doc_ids(line) if not own_doc_id or item.upper() != own_doc_id.upper()]
        article_labels = extract_article_labels(line)
        relation_type = _relation_type(line)
        if not doc_ids and not article_labels:
            continue
        for doc_id in doc_ids or [""]:
            rows.append(
                {
                    "target_doc_id": doc_id,
                    "target_article_label": article_labels[0] if article_labels else "",
                    "relation_type": relation_type,
                    "evidence_text": normalize_text(line)[:300],
                }
            )
    return _dedupe_dicts(rows)


def _relation_type(text: str) -> str:
    lowered = text.lower()
    if _contains_any(lowered, GUIDES_MARKERS):
        return "guides"
    if _contains_any(lowered, AMENDS_MARKERS):
        return "amends"
    if _contains_any(lowered, REPLACES_MARKERS):
        return "replaces"
    if _contains_any(lowered, ABOLISHES_MARKERS):
        return "abolishes"
    if "theo quy định tại" in lowered or "căn cứ" in lowered:
        return "references"
    return "references"


def _infer_norm_roles(article_title: str, article_text: str, doc_relations: dict[str, Any]) -> list[str]:
    lowered = normalize_text(f"{article_title}\n{article_text}").lower()
    roles: list[str] = []
    if any(term in lowered for term in ["hồ sơ", "thủ tục", "trình tự", "thời hạn", "quy trình"]):
        roles.append("procedure")
    if any(term in lowered for term in ["thẩm quyền", "cơ quan", "ủy ban", "bộ", "chính phủ"]):
        roles.append("authority")
    if any(term in lowered for term in ["xử phạt", "mức phạt", "phạt tiền"]):
        roles.append("penalty")
    if any(term in lowered for term in ["khắc phục hậu quả", "biện pháp khắc phục"]):
        roles.append("remedy")
    if any(term in lowered for term in ["điều kiện", "tiêu chí", "trường hợp"]):
        roles.append("condition")
    if any(term in lowered for term in ["hỗ trợ", "ưu đãi", "miễn", "giảm"]):
        roles.append("support_policy")
    if any(term in lowered for term in ["hiệu lực thi hành", "có hiệu lực"]) and "effective" not in roles:
        roles.append("effective")
    if any(term in lowered for term in ["điều khoản thi hành", "tổ chức thực hiện"]) and "implementation" not in roles:
        roles.append("implementation")
    if any(term in lowered for term in ["trách nhiệm", "chịu trách nhiệm"]) and "responsibility" not in roles:
        roles.append("responsibility")
    if any(term in lowered for term in ["chuyển tiếp"]) and "transition" not in roles:
        roles.append("transition")
    if any(term in lowered for term in ["sửa đổi", "bổ sung", "bãi bỏ", "thay thế"]) and "amendment" not in roles:
        roles.append("amendment")
    if doc_relations.get("guides_doc_ids") and "guidance" not in roles:
        roles.append("guidance")
    return roles or ["content"]


def _graph_neighbors(relations: dict[str, Any]) -> list[str]:
    output = []
    for key in ["guides_doc_ids", "amends_doc_ids", "replaces_doc_ids"]:
        output.extend(relations.get(key, []))
    return _dedupe(output)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _last_match_text(pattern: re.Pattern[str], text: str) -> str:
    matches = list(pattern.finditer(text))
    if not matches:
        return ""
    return normalize_text(matches[-1].group(1))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.upper()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _dedupe_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("target_doc_id", "")).upper(),
            str(row.get("target_article_label", "")).lower(),
            str(row.get("relation_type", "")).lower(),
            str(row.get("evidence_text", "")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output
