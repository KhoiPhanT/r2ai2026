from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_rag.schemas.models import (
    FinalPrediction,
    GoldQuestionMetadata,
    PredictedQuestionMetadata,
    Question,
    QuestionRunTrace,
)


def load_gold_annotations(path: str | Path) -> tuple[dict[int, GoldQuestionMetadata], dict[int, set[str]], dict[int, set[str]]]:
    rows = _load_rows(path)
    metadata_by_id: dict[int, GoldQuestionMetadata] = {}
    docs_by_id: dict[int, set[str]] = {}
    articles_by_id: dict[int, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = int(row.get("question_id") or row.get("id") or 0)
        if not qid:
            continue
        metadata = GoldQuestionMetadata.from_dict(row)
        metadata_by_id[qid] = metadata
        docs = row.get("gold_relevant_docs", row.get("relevant_docs", [])) or []
        articles = row.get("gold_relevant_articles", row.get("relevant_articles", [])) or []
        docs_by_id[qid] = {str(item) for item in docs if str(item).strip()}
        articles_by_id[qid] = {str(item) for item in articles if str(item).strip()}
    return metadata_by_id, docs_by_id, articles_by_id


def compute_metadata_metrics(
    predicted_by_id: dict[int, PredictedQuestionMetadata],
    gold_by_id: dict[int, GoldQuestionMetadata],
) -> dict[str, Any]:
    fields = ["intent", "question_type", "answer_shape", "needs_guidance_docs"]
    correct = {field: 0 for field in fields}
    total = {field: 0 for field in fields}
    rows: list[dict[str, Any]] = []
    for qid, gold in sorted(gold_by_id.items()):
        predicted = predicted_by_id.get(qid)
        if predicted is None:
            continue
        row = {"id": qid}
        for field in fields:
            gold_value = getattr(gold, field)
            if gold_value in {"", None}:
                continue
            total[field] += 1
            predicted_value = getattr(predicted, field)
            matched = predicted_value == gold_value
            correct[field] += int(matched)
            row[field] = {"gold": gold_value, "predicted": predicted_value, "ok": matched}
        if len(row) > 1:
            rows.append(row)
    return {
        field: (correct[field] / total[field] if total[field] else None)
        for field in fields
    } | {"rows": rows, "questions": len(rows)}


def compute_prediction_metrics(
    predictions: dict[int, FinalPrediction],
    gold_docs_by_id: dict[int, set[str]],
    gold_articles_by_id: dict[int, set[str]],
    traces_by_id: dict[int, QuestionRunTrace] | None = None,
) -> dict[str, Any]:
    doc_counts = _empty_counts()
    article_counts = _empty_counts()
    citation_ok = 0
    supported_ok = 0
    hallucinations = 0
    minimality_ok = 0
    total = 0
    rows = []
    traces_by_id = traces_by_id or {}
    for qid, prediction in sorted(predictions.items()):
        gold_docs = gold_docs_by_id.get(qid, set())
        gold_articles = gold_articles_by_id.get(qid, set())
        if not gold_docs and not gold_articles:
            continue
        total += 1
        pred_docs = set(prediction.relevant_docs)
        pred_articles = set(prediction.relevant_articles)
        _accumulate_counts(doc_counts, pred_docs, gold_docs)
        _accumulate_counts(article_counts, pred_articles, gold_articles)
        trace = traces_by_id.get(qid)
        issues = trace.verifier_issues if trace else []
        citation_ok += int(not any(issue.startswith("citation_") or issue == "answer_missing_article_citation" for issue in issues))
        supported_ok += int(not issues)
        hallucinations += int(any(issue.startswith("citation_not_") or issue.startswith("used_evidence_not_retrieved") for issue in issues))
        minimality_ok += int(not any(issue.startswith("used_evidence_not_cited") for issue in issues))
        rows.append(
            {
                "id": qid,
                "doc_f2": _f_beta(_counts_for_sets(pred_docs, gold_docs), beta=2.0),
                "article_f2": _f_beta(_counts_for_sets(pred_articles, gold_articles), beta=2.0),
                "verifier_issues": issues,
            }
        )
    return {
        "questions": total,
        "doc_precision": _precision(doc_counts),
        "doc_recall": _recall(doc_counts),
        "doc_f2": _f_beta(doc_counts, beta=2.0),
        "article_precision": _precision(article_counts),
        "article_recall": _recall(article_counts),
        "article_f2": _f_beta(article_counts, beta=2.0),
        "citation_accuracy": (citation_ok / total if total else None),
        "supported_answer_ratio": (supported_ok / total if total else None),
        "hallucination_rate": (hallucinations / total if total else None),
        "evidence_minimality_rate": (minimality_ok / total if total else None),
        "rows": rows,
    }


def build_prediction_map(predictions: list[FinalPrediction]) -> dict[int, FinalPrediction]:
    return {prediction.id: prediction for prediction in predictions}


def build_trace_map(traces: list[QuestionRunTrace]) -> dict[int, QuestionRunTrace]:
    return {trace.id: trace for trace in traces}


def build_gold_scaffolds(
    questions: list[Question],
    tune_questions: list[Question],
    holdout_questions: list[Question],
    predicted_by_id: dict[int, PredictedQuestionMetadata],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    question_by_id = {question.id: question for question in questions}
    tune_rows = [
        {
            **question_by_id[question.id].to_dict(),
            **GoldQuestionMetadata.from_dict({"question_id": question.id}).to_dict(),
            **_scaffold_row(question_by_id[question.id], predicted_by_id[question.id]),
        }
        for question in tune_questions
    ]
    holdout_rows = [
        {
            **question_by_id[question.id].to_dict(),
            **GoldQuestionMetadata.from_dict({"question_id": question.id}).to_dict(),
            **_scaffold_row(question_by_id[question.id], predicted_by_id[question.id]),
        }
        for question in holdout_questions
    ]
    return tune_rows, holdout_rows


def write_jsonl_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _scaffold_row(question: Question, predicted: PredictedQuestionMetadata) -> dict[str, Any]:
    metadata = GoldQuestionMetadata(
        question_id=question.id,
        intent=predicted.intent,
        question_type=predicted.question_type,
        answer_shape=predicted.answer_shape,
        needs_guidance_docs=predicted.needs_guidance_docs,
        gold_relevant_docs=[],
        gold_relevant_articles=[],
        notes=f"predicted_facets={','.join(predicted.legal_facets[:4])}; retrieval_bias={predicted.retrieval_bias}",
    )
    return metadata.to_dict()


def _load_rows(path: str | Path) -> list[Any]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return []


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def _counts_for_sets(predicted: set[str], gold: set[str]) -> dict[str, int]:
    counts = _empty_counts()
    _accumulate_counts(counts, predicted, gold)
    return counts


def _accumulate_counts(counts: dict[str, int], predicted: set[str], gold: set[str]) -> None:
    counts["tp"] += len(predicted & gold)
    counts["fp"] += len(predicted - gold)
    counts["fn"] += len(gold - predicted)


def _precision(counts: dict[str, int]) -> float | None:
    denom = counts["tp"] + counts["fp"]
    return counts["tp"] / denom if denom else None


def _recall(counts: dict[str, int]) -> float | None:
    denom = counts["tp"] + counts["fn"]
    return counts["tp"] / denom if denom else None


def _f_beta(counts: dict[str, int], beta: float) -> float | None:
    precision = _precision(counts)
    recall = _recall(counts)
    if precision is None or recall is None:
        return None
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
