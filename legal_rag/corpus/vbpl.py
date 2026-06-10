from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from legal_rag.parser import parse_document
from legal_rag.parser.legal_text import _document_relations, _graph_neighbors, _infer_norm_roles
from legal_rag.schemas.models import ArticleNode, DocumentRecord
from legal_rag.utils.text import normalize_text

DEFAULT_INCLUDE_DOC_TYPES = {
    "Luật",
    "Bộ luật",
    "Nghị định",
    "Thông tư",
    "Thông tư liên tịch",
    "Nghị quyết",
    "Văn bản hợp nhất",
    "Pháp lệnh",
    "Lệnh",
}
DEFAULT_EXCLUDE_DOC_TYPES = {
    "Bản dịch văn bản",
    "Công văn",
    "Chỉ thị",
}
GENERATED_DATA_DIRS = ("law_data_normalized", "normalized", "indices")


@dataclass(slots=True)
class VbplImportConfig:
    include_doc_types: set[str] = field(default_factory=lambda: set(DEFAULT_INCLUDE_DOC_TYPES))
    exclude_doc_types: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDE_DOC_TYPES))
    include_decisions: bool = False
    include_english_like: bool = False


@dataclass(slots=True)
class VbplDocumentInfo:
    document_id: str
    document_number: str
    document_type: str
    source_document_type: str
    title: str
    summary: str
    issuer: str
    issue_date: str
    effective_date: str
    expiration_date: str
    legal_status: str
    source_url: str
    html_url: str
    language: str
    article_count: int
    clause_count: int
    point_count: int
    chunk_count: int
    text_hash: str
    text_length: int
    english_like: bool
    line_no: int = 0

    @property
    def title_for_submission(self) -> str:
        title = normalize_text(self.title)
        if self.document_number and self.document_number in title:
            return title
        pieces = [self.document_type, self.document_number, title]
        return normalize_text(" ".join(piece for piece in pieces if piece))

    @property
    def trich_yeu(self) -> str:
        summary = normalize_text(self.summary)
        if summary:
            return summary
        title = normalize_text(self.title)
        for token in (self.document_type, self.document_number, "số"):
            if token:
                title = re.sub(rf"\b{re.escape(token)}\b", " ", title, flags=re.IGNORECASE)
        return normalize_text(title) or self.title_for_submission

    def to_document_record(self, raw_text: str, relations: dict[str, Any] | None = None) -> DocumentRecord:
        metadata = {
            "source_dataset": "vbpl",
            "source_document_id": self.document_id,
            "source_document_type": self.source_document_type,
            "html_url": self.html_url,
            "expiration_date": self.expiration_date,
            "language": self.language,
            "article_count": self.article_count,
            "clause_count": self.clause_count,
            "point_count": self.point_count,
            "chunk_count": self.chunk_count,
            "text_hash": self.text_hash,
            "english_like": self.english_like,
        }
        if relations:
            metadata["document_relations"] = relations
            metadata["graph_neighbors"] = _graph_neighbors(relations)
        return DocumentRecord(
            doc_id=self.document_number,
            doc_type=self.document_type,
            trich_yeu=self.trich_yeu,
            title_for_submission=self.title_for_submission,
            raw_text=normalize_text(raw_text),
            issuer=self.issuer,
            issue_date=self.issue_date,
            effective_date=self.effective_date,
            status=self.legal_status,
            source_url=self.source_url or self.html_url,
            metadata=metadata,
        )


@dataclass(slots=True)
class VbplInspectReport:
    input: str
    documents: dict[str, Any]
    legal_units: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VbplImportReport:
    input: str
    output: str
    selected_documents: int
    written_documents: int
    written_articles: int
    written_legal_units: int
    fallback_articles: int
    excluded_documents: dict[str, int]
    duplicate_doc_numbers: list[dict[str, Any]]
    variant_conflicts: list[dict[str, Any]]
    parse_errors: list[str]
    missing_metadata: dict[str, int]
    doc_type_counts: dict[str, int]
    unit_type_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_vbpl_corpus(input_dir: str | Path, report_path: str | Path) -> VbplInspectReport:
    root = Path(input_dir)
    documents_path = root / "documents.jsonl"
    units_path = root / "legal_units.jsonl"
    document_report = _inspect_documents(documents_path)
    units_report = _inspect_units(units_path)
    report = VbplInspectReport(str(root), document_report, units_report)
    _write_json(report_path, report.to_dict())
    return report


def import_vbpl_corpus(
    input_dir: str | Path,
    output_dir: str | Path,
    config: VbplImportConfig | None = None,
) -> VbplImportReport:
    cfg = config or VbplImportConfig()
    root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    documents_path = root / "documents.jsonl"
    units_path = root / "legal_units.jsonl"
    if not documents_path.exists():
        raise FileNotFoundError(documents_path)
    if not units_path.exists():
        raise FileNotFoundError(units_path)

    doc_infos, doc_scan = _load_document_infos(documents_path)
    selected_by_number, duplicate_rows, variant_conflicts, excluded = _select_canonical_documents(doc_infos, cfg)
    selected_ids = {info.document_id for info in selected_by_number.values()}
    relations_by_id: dict[str, dict[str, Any]] = {}
    raw_text_by_id_for_fallback: dict[str, str] = {}
    written_documents = _write_selected_documents(
        documents_path,
        output_root / "documents.jsonl",
        selected_ids,
        doc_infos,
        relations_by_id,
        raw_text_by_id_for_fallback,
    )

    articles_path = output_root / "articles.jsonl"
    compact_units_path = output_root / "legal_units_compact.jsonl"
    written_articles, written_units, unit_type_counts, docs_with_units = _write_articles_from_units(
        units_path,
        articles_path,
        compact_units_path,
        selected_ids,
        doc_infos,
        relations_by_id,
    )

    fallback_articles = _append_fallback_articles(
        articles_path,
        selected_ids - docs_with_units,
        doc_infos,
        raw_text_by_id_for_fallback,
        relations_by_id,
    )
    written_articles += fallback_articles

    report = VbplImportReport(
        input=str(root),
        output=str(output_root),
        selected_documents=len(selected_ids),
        written_documents=written_documents,
        written_articles=written_articles,
        written_legal_units=written_units,
        fallback_articles=fallback_articles,
        excluded_documents=dict(excluded),
        duplicate_doc_numbers=duplicate_rows[:200],
        variant_conflicts=variant_conflicts[:200],
        parse_errors=doc_scan["parse_errors"][:200],
        missing_metadata=dict(doc_scan["missing_metadata"]),
        doc_type_counts=dict(doc_scan["doc_type_counts"]),
        unit_type_counts=dict(unit_type_counts),
    )
    _write_json(output_root / "vbpl_import_report.json", report.to_dict())
    return report


def clean_generated_data(data_root: str | Path = "data") -> list[str]:
    root = Path(data_root)
    removed: list[str] = []
    for dirname in GENERATED_DATA_DIRS:
        path = root / dirname
        _ensure_generated_path(root, path)
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, None, f"{path}:{line_no}:{exc}"
                continue
            if not isinstance(data, dict):
                yield line_no, None, f"{path}:{line_no}:not_object"
                continue
            yield line_no, data, None


def _inspect_documents(path: Path) -> dict[str, Any]:
    doc_type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    duplicate_numbers: Counter[str] = Counter()
    english_like = 0
    parse_errors: list[str] = []
    lines = 0
    valid = 0
    article_total = clause_total = point_total = 0
    for line_no, raw, error in _iter_jsonl(path):
        lines = line_no
        if error:
            parse_errors.append(error)
            continue
        assert raw is not None
        valid += 1
        for field in ["document_id", "document_number", "document_type", "title", "full_text"]:
            if not raw.get(field):
                missing[field] += 1
        doc_type = normalize_text(str(raw.get("document_type") or ""))
        doc_type_counts[doc_type] += 1
        status_counts[normalize_text(str(raw.get("legal_status") or ""))] += 1
        language_counts[normalize_text(str(raw.get("language") or ""))] += 1
        number = normalize_text(str(raw.get("document_number") or ""))
        if number:
            duplicate_numbers[number] += 1
        article_total += _safe_int(raw.get("article_count"))
        clause_total += _safe_int(raw.get("clause_count"))
        point_total += _safe_int(raw.get("point_count"))
        if doc_type == "Bản dịch văn bản" or _looks_english(str(raw.get("title") or ""), str(raw.get("full_text") or "")[:2000]):
            english_like += 1
    duplicate_preview = [
        {"document_number": number, "count": count}
        for number, count in duplicate_numbers.most_common()
        if count > 1
    ][:100]
    return {
        "path": str(path),
        "lines": lines,
        "valid_records": valid,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:100],
        "doc_type_counts": dict(doc_type_counts),
        "status_counts": dict(status_counts),
        "language_counts": dict(language_counts),
        "missing_metadata": dict(missing),
        "duplicate_document_numbers": duplicate_preview,
        "english_like_documents": english_like,
        "article_count": article_total,
        "clause_count": clause_total,
        "point_count": point_total,
    }


def _inspect_units(path: Path) -> dict[str, Any]:
    unit_type_counts: Counter[str] = Counter()
    doc_ids: set[str] = set()
    article_keys: set[tuple[str, str]] = set()
    missing: Counter[str] = Counter()
    parse_errors: list[str] = []
    lines = 0
    valid = 0
    for line_no, raw, error in _iter_jsonl(path):
        lines = line_no
        if error:
            parse_errors.append(error)
            continue
        assert raw is not None
        valid += 1
        for field in ["document_id", "document_number", "document_type", "article_no", "unit_type", "text"]:
            if not raw.get(field):
                missing[field] += 1
        doc_id = str(raw.get("document_id") or "")
        if doc_id:
            doc_ids.add(doc_id)
        article_no = str(raw.get("article_no") or "")
        if doc_id and article_no:
            article_keys.add((doc_id, article_no))
        unit_type_counts[normalize_text(str(raw.get("unit_type") or ""))] += 1
    return {
        "path": str(path),
        "lines": lines,
        "valid_records": valid,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:100],
        "document_count": len(doc_ids),
        "article_key_count": len(article_keys),
        "unit_type_counts": dict(unit_type_counts),
        "missing_metadata": dict(missing),
    }


def _load_document_infos(path: Path) -> tuple[dict[str, VbplDocumentInfo], dict[str, Any]]:
    infos: dict[str, VbplDocumentInfo] = {}
    doc_type_counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    parse_errors: list[str] = []
    for line_no, raw, error in _iter_jsonl(path):
        if error:
            parse_errors.append(error)
            continue
        assert raw is not None
        info = _document_info_from_raw(raw, line_no)
        if not info.document_id:
            missing["document_id"] += 1
            continue
        for field in ["document_number", "document_type", "title", "full_text"]:
            if not raw.get(field):
                missing[field] += 1
        doc_type_counts[info.document_type] += 1
        infos[info.document_id] = info
    return infos, {"parse_errors": parse_errors, "missing_metadata": missing, "doc_type_counts": doc_type_counts}


def _document_info_from_raw(raw: dict[str, Any], line_no: int) -> VbplDocumentInfo:
    full_text = str(raw.get("full_text") or "")
    title = normalize_text(str(raw.get("title") or ""))
    source_document_type = normalize_text(str(raw.get("document_type") or ""))
    language = normalize_text(str(raw.get("language") or ""))
    english_like = _looks_english(title, full_text[:2000])
    document_type = source_document_type
    if source_document_type == "Bản dịch văn bản" and language.lower().startswith("vi"):
        document_type = _document_type_from_vietnamese_title(title) or source_document_type
    return VbplDocumentInfo(
        document_id=normalize_text(str(raw.get("document_id") or "")),
        document_number=normalize_text(str(raw.get("document_number") or "")),
        document_type=document_type,
        source_document_type=source_document_type,
        title=title,
        summary=normalize_text(str(raw.get("summary") or "")),
        issuer=normalize_text(str(raw.get("issuing_agency") or "")),
        issue_date=normalize_text(str(raw.get("issue_date") or "")),
        effective_date=normalize_text(str(raw.get("effective_date") or "")),
        expiration_date=normalize_text(str(raw.get("expiration_date") or "")),
        legal_status=normalize_text(str(raw.get("legal_status") or "")),
        source_url=normalize_text(str(raw.get("source_url") or "")),
        html_url=normalize_text(str(raw.get("html_url") or "")),
        language=language,
        article_count=_safe_int(raw.get("article_count")),
        clause_count=_safe_int(raw.get("clause_count")),
        point_count=_safe_int(raw.get("point_count")),
        chunk_count=_safe_int(raw.get("chunk_count")),
        text_hash=_hash_text(full_text),
        text_length=len(full_text),
        english_like=english_like,
        line_no=line_no,
    )


def _document_type_from_vietnamese_title(title: str) -> str:
    for document_type in sorted(DEFAULT_INCLUDE_DOC_TYPES, key=len, reverse=True):
        if re.match(rf"^{re.escape(document_type)}(?:\s|$)", title, flags=re.IGNORECASE):
            return document_type
    return ""


def _select_canonical_documents(
    docs: dict[str, VbplDocumentInfo],
    config: VbplImportConfig,
) -> tuple[dict[str, VbplDocumentInfo], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    grouped: dict[tuple[str, str], list[VbplDocumentInfo]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    for info in docs.values():
        reason = _exclusion_reason(info, config)
        if reason:
            excluded[reason] += 1
            continue
        grouped[(info.document_number, info.document_type)].append(info)

    selected: dict[str, VbplDocumentInfo] = {}
    duplicate_rows: list[dict[str, Any]] = []
    variant_conflicts: list[dict[str, Any]] = []
    for (document_number, document_type), variants in grouped.items():
        variants.sort(key=_canonical_rank, reverse=True)
        selected[f"{document_number}|{document_type}"] = variants[0]
        if len(variants) > 1:
            hashes = sorted({item.text_hash for item in variants if item.text_hash})
            row = {
                "document_number": document_number,
                "document_type": document_type,
                "count": len(variants),
                "selected_document_id": variants[0].document_id,
                "text_hashes": hashes[:10],
            }
            duplicate_rows.append(row)
            if len(hashes) > 1:
                variant_conflicts.append(row)
    return selected, duplicate_rows, variant_conflicts, excluded


def _exclusion_reason(info: VbplDocumentInfo, config: VbplImportConfig) -> str:
    if not info.document_number:
        return "missing_document_number"
    if not info.document_type:
        return "missing_document_type"
    if info.document_type in config.exclude_doc_types:
        return f"excluded_doc_type:{info.document_type}"
    if info.document_type == "Quyết định" and not config.include_decisions:
        return "excluded_doc_type:Quyết định"
    if info.document_type not in config.include_doc_types and not (info.document_type == "Quyết định" and config.include_decisions):
        return f"not_in_include_doc_types:{info.document_type}"
    if not config.include_english_like and (info.document_type == "Bản dịch văn bản" or info.english_like):
        return "english_like_or_translation"
    if info.article_count <= 0 and info.text_length <= 0:
        return "empty_document"
    return ""


def _canonical_rank(info: VbplDocumentInfo) -> tuple[int, int, int, int, int]:
    current = 1 if "còn hiệu lực" in info.legal_status.lower() else 0
    vietnamese = 0 if info.english_like or info.document_type == "Bản dịch văn bản" else 1
    consolidated = 1 if info.document_type == "Văn bản hợp nhất" else 0
    structure = info.article_count + info.clause_count + info.point_count
    title_len = len(info.title)
    return vietnamese, current, consolidated, structure, title_len


def _write_selected_documents(
    documents_path: Path,
    output_path: Path,
    selected_ids: set[str],
    doc_infos: dict[str, VbplDocumentInfo],
    relations_by_id: dict[str, dict[str, Any]],
    raw_text_by_id_for_fallback: dict[str, str],
) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for _line_no, raw, error in _iter_jsonl(documents_path):
            if error or raw is None:
                continue
            doc_id = normalize_text(str(raw.get("document_id") or ""))
            if doc_id not in selected_ids:
                continue
            info = doc_infos[doc_id]
            raw_text = normalize_text(str(raw.get("full_text") or ""))
            record_without_relations = info.to_document_record(raw_text)
            relations = _compact_relations(_document_relations(record_without_relations, _relation_scope(raw_text)))
            relations_by_id[doc_id] = relations
            raw_text_by_id_for_fallback[doc_id] = raw_text
            record = info.to_document_record(raw_text, relations)
            out.write(json.dumps(_document_record_to_dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_articles_from_units(
    units_path: Path,
    articles_path: Path,
    compact_units_path: Path,
    selected_ids: set[str],
    doc_infos: dict[str, VbplDocumentInfo],
    relations_by_id: dict[str, dict[str, Any]],
) -> tuple[int, int, Counter[str], set[str]]:
    articles_path.parent.mkdir(parents=True, exist_ok=True)
    written_articles = 0
    written_units = 0
    unit_type_counts: Counter[str] = Counter()
    docs_with_units: set[str] = set()
    current_key: tuple[str, str] | None = None
    current_rows: list[dict[str, Any]] = []
    seen_article_keys: set[tuple[str, str]] = set()
    written_article_keys: set[str] = set()
    non_contiguous: set[tuple[str, str]] = set()

    with articles_path.open("w", encoding="utf-8") as article_out, compact_units_path.open("w", encoding="utf-8") as unit_out:
        for _line_no, raw, error in _iter_jsonl(units_path):
            if error or raw is None:
                continue
            doc_id = normalize_text(str(raw.get("document_id") or ""))
            if doc_id not in selected_ids:
                continue
            article_no = normalize_text(str(raw.get("article_no") or ""))
            if not article_no:
                continue
            key = (doc_id, article_no)
            if current_key is not None and key != current_key:
                article = _article_from_unit_rows(current_rows, doc_infos[current_key[0]], relations_by_id.get(current_key[0], {}))
                if article is not None and article.article_key not in written_article_keys:
                    article_out.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
                    written_article_keys.add(article.article_key)
                    written_articles += 1
                    docs_with_units.add(current_key[0])
                seen_article_keys.add(current_key)
                current_rows = []
            if key in seen_article_keys and key != current_key:
                non_contiguous.add(key)
            current_key = key
            current_rows.append(raw)
            unit_type = normalize_text(str(raw.get("unit_type") or ""))
            unit_type_counts[unit_type] += 1
            unit_out.write(json.dumps(_compact_unit(raw), ensure_ascii=False) + "\n")
            written_units += 1

        if current_key is not None and current_rows:
            article = _article_from_unit_rows(current_rows, doc_infos[current_key[0]], relations_by_id.get(current_key[0], {}))
            if article is not None and article.article_key not in written_article_keys:
                article_out.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
                written_article_keys.add(article.article_key)
                written_articles += 1
                docs_with_units.add(current_key[0])
    if non_contiguous:
        unit_type_counts["non_contiguous_article_keys"] = len(non_contiguous)
    return written_articles, written_units, unit_type_counts, docs_with_units


def _article_from_unit_rows(
    rows: list[dict[str, Any]],
    info: VbplDocumentInfo,
    document_relations: dict[str, Any],
) -> ArticleNode | None:
    if not rows:
        return None
    article_no = normalize_text(str(rows[0].get("article_no") or ""))
    if not article_no:
        return None
    article_label = f"Điều {article_no}"
    article_title = normalize_text(str(rows[0].get("article_title") or ""))
    article_order = _safe_int(article_no)
    article_text = _assemble_article_text(rows, article_label, article_title)
    if not article_text:
        return None
    clause_nodes = _clause_nodes(rows, article_label)
    references = []
    record = info.to_document_record(article_text, document_relations)
    norm_roles = _infer_norm_roles(article_title, article_text, document_relations)
    metadata = dict(record.metadata)
    metadata.update(
        {
            "source_dataset": "vbpl",
            "chunk_type": "article",
            "node_type": "article",
            "doc_id": info.document_number,
            "article_label": article_label,
            "article_no": article_no,
            "article_order": article_order,
            "law_title": info.title_for_submission,
            "part_label": _last_nonempty(rows, "part_no", prefix="Phần "),
            "part_title": _last_nonempty(rows, "part_title"),
            "chapter_label": _last_nonempty(rows, "chapter_no", prefix="Chương "),
            "chapter_title": _last_nonempty(rows, "chapter_title"),
            "section_label": _last_nonempty(rows, "section_no", prefix="Mục "),
            "section_title": _last_nonempty(rows, "section_title"),
            "clause_nodes": clause_nodes,
            "references": references or document_relations.get("references", []),
            "norm_roles": norm_roles,
            "primary_norm_role": norm_roles[0] if norm_roles else "content",
        }
    )
    if document_relations:
        metadata["document_relations"] = document_relations
        metadata["graph_neighbors"] = _graph_neighbors(document_relations)
    article_key = f"{info.document_number}|{info.title_for_submission}|{article_label}"
    return ArticleNode(
        article_key=article_key,
        doc_id=info.document_number,
        doc_type=info.document_type,
        title_for_submission=info.title_for_submission,
        article_label=article_label,
        article_title=article_title,
        text=article_text,
        status=info.legal_status,
        source_url=info.source_url or info.html_url,
        metadata=metadata,
    )


def _assemble_article_text(rows: list[dict[str, Any]], article_label: str, article_title: str) -> str:
    article_rows = [row for row in rows if str(row.get("unit_type") or "") == "article"]
    article_text = normalize_text(str(article_rows[0].get("text") or "")) if article_rows else ""
    if not article_text:
        article_text = normalize_text(f"{article_label}. {article_title}")
    elif not article_text.lower().startswith(article_label.lower()):
        article_text = normalize_text(f"{article_label}. {article_title}\n{article_text}")
    pieces = [article_text]
    for clause_no in _ordered_values(row.get("clause_no") for row in rows if row.get("clause_no")):
        clause_row_texts = [
            normalize_text(str(row.get("text") or ""))
            for row in rows
            if normalize_text(str(row.get("clause_no") or "")) == clause_no and str(row.get("unit_type") or "") == "clause"
        ]
        for text in clause_row_texts:
            _append_unique(pieces, text)
        point_rows = [
            row
            for row in rows
            if normalize_text(str(row.get("clause_no") or "")) == clause_no and str(row.get("unit_type") or "") == "point"
        ]
        for row in sorted(point_rows, key=lambda item: _sort_key(str(item.get("point_no") or ""))):
            _append_unique(pieces, normalize_text(str(row.get("text") or "")))
    loose_rows = [row for row in rows if str(row.get("unit_type") or "") not in {"article", "clause", "point"}]
    for row in loose_rows:
        _append_unique(pieces, normalize_text(str(row.get("text") or "")))
    return normalize_text("\n".join(piece for piece in pieces if piece))


def _clause_nodes(rows: list[dict[str, Any]], article_label: str) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for clause_no in _ordered_values(row.get("clause_no") for row in rows if row.get("clause_no")):
        clause_label = f"Khoản {clause_no}"
        clause_texts = [
            normalize_text(str(row.get("text") or ""))
            for row in rows
            if normalize_text(str(row.get("clause_no") or "")) == clause_no and str(row.get("unit_type") or "") == "clause"
        ]
        point_nodes = []
        for row in sorted(
            [
                item
                for item in rows
                if normalize_text(str(item.get("clause_no") or "")) == clause_no and str(item.get("unit_type") or "") == "point"
            ],
            key=lambda item: _sort_key(str(item.get("point_no") or "")),
        ):
            point_no = normalize_text(str(row.get("point_no") or ""))
            point_label = f"Điểm {point_no}" if point_no else "Điểm"
            point_text = normalize_text(str(row.get("text") or ""))
            if not point_text:
                continue
            point_nodes.append(
                {
                    "span_key": f"{article_label}|{clause_label}|{point_label}",
                    "parent_span_key": f"{article_label}|{clause_label}",
                    "level": "point",
                    "label": point_label,
                    "text": point_text,
                }
            )
        clause_text = normalize_text("\n".join([*clause_texts, *(point["text"] for point in point_nodes)]))
        if not clause_text:
            continue
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


def _append_fallback_articles(
    articles_path: Path,
    doc_ids: set[str],
    doc_infos: dict[str, VbplDocumentInfo],
    raw_text_by_id: dict[str, str],
    relations_by_id: dict[str, dict[str, Any]],
) -> int:
    if not doc_ids:
        return 0
    fallback_articles: list[ArticleNode] = []
    for doc_id in sorted(doc_ids):
        info = doc_infos.get(doc_id)
        raw_text = raw_text_by_id.get(doc_id, "")
        if info is None or not raw_text:
            continue
        record = info.to_document_record(raw_text, relations_by_id.get(doc_id, {}))
        fallback_articles.extend(parse_document(record))
    if not fallback_articles:
        return 0
    existing_keys = _article_keys_in_file(articles_path)
    written = 0
    with articles_path.open("a", encoding="utf-8") as out:
        for article in fallback_articles:
            if article.article_key in existing_keys:
                continue
            out.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
            existing_keys.add(article.article_key)
            written += 1
    return written


def _article_keys_in_file(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("article_key") or "")
            if key:
                keys.add(key)
    return keys


def _document_record_to_dict(record: DocumentRecord) -> dict[str, Any]:
    return {
        "doc_id": record.doc_id,
        "doc_type": record.doc_type,
        "trich_yeu": record.trich_yeu,
        "title_for_submission": record.title_for_submission,
        "raw_text": record.raw_text,
        "issuer": record.issuer,
        "issue_date": record.issue_date,
        "effective_date": record.effective_date,
        "status": record.status,
        "source_url": record.source_url,
        **record.metadata,
    }


def _relation_scope(raw_text: str, max_lines: int = 120, max_chars: int = 20000) -> str:
    lines = raw_text.splitlines()[:max_lines]
    return "\n".join(lines)[:max_chars]


def _compact_relations(relations: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    return {
        "guides_doc_ids": list(relations.get("guides_doc_ids", []))[:limit],
        "amends_doc_ids": list(relations.get("amends_doc_ids", []))[:limit],
        "replaces_doc_ids": list(relations.get("replaces_doc_ids", []))[:limit],
        "abolishes_doc_ids": list(relations.get("abolishes_doc_ids", []))[:limit],
        "references_only": list(relations.get("references_only", []))[:limit],
        "references": list(relations.get("references", []))[:limit],
    }


def _compact_unit(raw: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "unit_id",
        "unit_type",
        "document_id",
        "document_number",
        "document_type",
        "document_title",
        "part_no",
        "part_title",
        "chapter_no",
        "chapter_title",
        "section_no",
        "section_title",
        "article_no",
        "article_title",
        "clause_no",
        "point_no",
        "source_url",
        "text",
    ]
    return {key: raw.get(key, "") for key in keys}


def _ordered_values(values: Iterable[Any]) -> list[str]:
    unique = {normalize_text(str(value or "")) for value in values if normalize_text(str(value or ""))}
    return sorted(unique, key=_sort_key)


def _sort_key(value: str) -> tuple[int, str]:
    text = normalize_text(value)
    if text.isdigit():
        return int(text), ""
    match = re.match(r"^(\d+)([A-Za-zĐđ])$", text)
    if match:
        return int(match.group(1)), match.group(2).lower()
    return 10**9, text.lower()


def _append_unique(pieces: list[str], text: str) -> None:
    if not text:
        return
    if any(text == piece or text in piece for piece in pieces):
        return
    pieces.append(text)


def _last_nonempty(rows: list[dict[str, Any]], key: str, prefix: str = "") -> str:
    for row in reversed(rows):
        value = normalize_text(str(row.get(key) or ""))
        if value:
            return f"{prefix}{value}" if prefix and not value.lower().startswith(prefix.strip().lower()) else value
    return ""


def _looks_english(title: str, sample: str) -> bool:
    body = (sample or "").strip()
    text = body if len(body) >= 200 else f"{title}\n{body}".strip()
    if not text:
        return False
    lowered = text[:3000].lower()
    english_markers = (" decree ", " law ", " circular ", " decision ", " article ", " government ", " ministry ")
    vietnamese_markers = (" luật ", " nghị định ", " thông tư ", " điều ", " khoản ", " điểm ", " chính phủ ")
    padded = f" {lowered} "
    english_hits = sum(padded.count(marker) for marker in english_markers)
    vietnamese_hits = sum(padded.count(marker) for marker in vietnamese_markers)
    if vietnamese_hits >= max(2, english_hits):
        return False
    ascii_letters = sum(1 for ch in lowered if "a" <= ch <= "z")
    vietnamese_chars = sum(1 for ch in lowered if ch in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    return english_hits >= 2 and ascii_letters > max(80, vietnamese_chars * 20)


def _hash_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_generated_path(root: Path, path: Path) -> None:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    raw_path = (root / "law_data_raw").resolve()
    if path_resolved == raw_path or raw_path in path_resolved.parents:
        raise ValueError(f"refuse_to_delete_raw_source:{path}")
    if path_resolved.parent != root_resolved:
        raise ValueError(f"refuse_to_delete_outside_data_root:{path}")
