from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from legal_rag.schemas.models import ArticleNode, FinalPrediction, Question
from legal_rag.utils.text import extract_article_labels

DOC_FIELD_PATTERN = re.compile(r"^[^|]+\|[^|]+$")
ARTICLE_FIELD_PATTERN = re.compile(r"^[^|]+\|[^|]+\|Điều\s+\d+[a-zA-Z]?$")


def format_prediction(question: Question, answer: str, articles: list[ArticleNode]) -> FinalPrediction:
    docs = _dedupe(article.relevant_doc for article in articles)
    relevant_articles = _dedupe(article.relevant_article for article in articles)
    return FinalPrediction(
        id=question.id,
        question=question.question,
        answer=answer,
        relevant_docs=docs,
        relevant_articles=relevant_articles,
    )


def write_predictions(predictions: list[FinalPrediction], output_path: str | Path) -> None:
    payload = [prediction.to_dict() for prediction in predictions]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions(path: str | Path) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Question(id=int(item["id"]), question=str(item["question"])) for item in data]


def validate_submission(input_path: str | Path, questions_path: str | Path | None = None) -> list[str]:
    issues: list[str] = []
    path = Path(input_path)
    if path.name != "results.json":
        issues.append("file_must_be_named_results.json")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"invalid_json:{exc}"]

    if not isinstance(data, list):
        return ["root_must_be_json_array"]

    expected_by_id: dict[int, str] | None = None
    if questions_path is not None:
        questions = load_questions(questions_path)
        expected_by_id = {q.id: q.question for q in questions}
        if len(data) != len(questions):
            issues.append(f"row_count_mismatch:got={len(data)} expected={len(questions)}")

    seen_ids: set[int] = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            issues.append(f"row_not_object:{idx}")
            continue
        for field in ["id", "question", "answer", "relevant_docs", "relevant_articles"]:
            if field not in item:
                issues.append(f"missing_field:{idx}:{field}")
        if "id" not in item:
            continue
        try:
            qid = int(item["id"])
        except Exception:  # noqa: BLE001
            issues.append(f"id_not_int:{idx}:{item.get('id')}")
            continue
        if qid in seen_ids:
            issues.append(f"duplicate_id:{qid}")
        seen_ids.add(qid)
        if expected_by_id is not None:
            expected_question = expected_by_id.get(qid)
            if expected_question is None:
                issues.append(f"unknown_id:{qid}")
            elif item.get("question") != expected_question:
                issues.append(f"question_mismatch:{qid}")
        _validate_row_fields(idx, item, issues)

    if expected_by_id is not None:
        missing = sorted(set(expected_by_id) - seen_ids)
        if missing:
            preview = ",".join(str(i) for i in missing[:20])
            issues.append(f"missing_ids:{preview}")
    return issues


def package_submission(input_path: str | Path, output_path: str | Path) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(input_file, arcname="results.json")


def validate_package_manifest(input_path: str | Path) -> list[str]:
    issues: list[str] = []
    manifest_path = Path(input_path).with_suffix(".manifest.json")
    if not manifest_path.exists():
        return [f"missing_manifest:{manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"invalid_manifest_json:{exc}"]

    backend = manifest.get("generator_backend")
    if backend != "ollama":
        issues.append(f"submission_requires_ollama_generator:got={backend}")
    if not manifest.get("model"):
        issues.append("manifest_missing_model")
    if not manifest.get("ollama_url"):
        issues.append("manifest_missing_ollama_url")
    verifier_issues = manifest.get("verifier_issues") or []
    if verifier_issues:
        issues.append(f"manifest_has_verifier_issues:{len(verifier_issues)}")
    return issues


def _validate_row_fields(idx: int, item: dict[str, Any], issues: list[str]) -> None:
    if not isinstance(item.get("question"), str) or not item.get("question"):
        issues.append(f"invalid_question:{idx}")
    if not isinstance(item.get("answer"), str) or not item.get("answer"):
        issues.append(f"invalid_answer:{idx}")
    if not isinstance(item.get("relevant_docs"), list):
        issues.append(f"relevant_docs_not_list:{idx}")
    else:
        for value in item["relevant_docs"]:
            if not isinstance(value, str) or not DOC_FIELD_PATTERN.match(value):
                issues.append(f"invalid_relevant_doc:{idx}:{value}")
    if not isinstance(item.get("relevant_articles"), list):
        issues.append(f"relevant_articles_not_list:{idx}")
    else:
        for value in item["relevant_articles"]:
            if not isinstance(value, str) or not ARTICLE_FIELD_PATTERN.match(value):
                issues.append(f"invalid_relevant_article:{idx}:{value}")

    if item.get("relevant_articles") and not extract_article_labels(item.get("answer", "")):
        issues.append(f"answer_missing_dieu_pattern:{idx}")


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
