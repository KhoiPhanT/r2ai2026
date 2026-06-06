from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from legal_rag.parser import parse_document
from legal_rag.schemas.models import ArticleNode, DocumentRecord
from legal_rag.utils.text import normalize_text

REQUIRED_FIELDS = {"doc_id", "doc_type", "trich_yeu", "title_for_submission", "raw_text"}


def ingest_corpus(input_path: str | Path) -> tuple[list[ArticleNode], list[str]]:
    records, warnings = load_document_records(input_path)
    articles: list[ArticleNode] = []
    for record in records:
        articles.extend(parse_document(record))
    return articles, warnings


def load_document_records(input_path: str | Path) -> tuple[list[DocumentRecord], list[str]]:
    path = Path(input_path)
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    records: list[DocumentRecord] = []
    warnings: list[str] = []

    for file_path in files:
        if file_path.name.startswith(".") or file_path.is_dir():
            continue
        if file_path.suffix.lower() in {".json", ".jsonl"}:
            raw_records = _load_json_records(file_path)
            for idx, raw in enumerate(raw_records):
                record, issues = _record_from_mapping(raw, f"{file_path}:{idx + 1}")
                warnings.extend(issues)
                if record is not None:
                    records.append(record)
        elif file_path.suffix.lower() in {".txt", ".md"}:
            records.append(_record_from_text_file(file_path))
        else:
            warnings.append(f"skip_unsupported_file:{file_path}")
    return records, warnings


def write_articles_jsonl(articles: Iterable[ArticleNode], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")


def read_articles_jsonl(input_path: str | Path) -> list[ArticleNode]:
    articles: list[ArticleNode] = []
    with Path(input_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(ArticleNode.from_dict(json.loads(line)))
    return articles


def _load_json_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("documents"), list):
        return data["documents"]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON corpus shape: {path}")


def _record_from_mapping(raw: dict, source: str) -> tuple[DocumentRecord | None, list[str]]:
    issues: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(raw))
    if missing:
        issues.append(f"missing_required_fields:{source}:{','.join(missing)}")
        return None, issues

    title = normalize_text(str(raw.get("title_for_submission", "")))
    doc_id = normalize_text(str(raw.get("doc_id", "")))
    doc_type = normalize_text(str(raw.get("doc_type", "")))
    trich_yeu = normalize_text(str(raw.get("trich_yeu", "")))

    if doc_id and doc_id not in title:
        issues.append(f"title_missing_doc_id:{source}:{doc_id}")
    if doc_type and not title.lower().startswith(doc_type.lower()):
        issues.append(f"title_missing_doc_type_prefix:{source}:{doc_type}")

    known = {
        "doc_id",
        "doc_type",
        "trich_yeu",
        "title_for_submission",
        "raw_text",
        "issuer",
        "issue_date",
        "effective_date",
        "status",
        "source_url",
    }
    metadata = {k: v for k, v in raw.items() if k not in known}
    return (
        DocumentRecord(
            doc_id=doc_id,
            doc_type=doc_type,
            trich_yeu=trich_yeu,
            title_for_submission=title,
            raw_text=normalize_text(str(raw.get("raw_text", ""))),
            issuer=normalize_text(str(raw.get("issuer", ""))),
            issue_date=normalize_text(str(raw.get("issue_date", ""))),
            effective_date=normalize_text(str(raw.get("effective_date", ""))),
            status=normalize_text(str(raw.get("status", ""))),
            source_url=normalize_text(str(raw.get("source_url", ""))),
            metadata=metadata,
        ),
        issues,
    )


def _record_from_text_file(path: Path) -> DocumentRecord:
    stem = path.stem
    return DocumentRecord(
        doc_id=stem,
        doc_type="Văn bản",
        trich_yeu=stem,
        title_for_submission=f"Văn bản {stem}",
        raw_text=path.read_text(encoding="utf-8"),
        source_url=str(path),
        metadata={"source_file": str(path), "metadata_inferred": True},
    )

