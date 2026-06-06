from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.corpus.ingest import ingest_corpus, read_articles_jsonl, write_articles_jsonl
from legal_rag.documents import normalize_documents
from legal_rag.formatting.submission import (
    format_prediction,
    load_questions,
    package_submission,
    validate_submission,
    write_predictions,
)
from legal_rag.generation import generate_grounded_answer
from legal_rag.retrieval import BM25Index, retrieve_articles
from legal_rag.verifier import verify_prediction_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="legal_rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest_corpus")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--output", required=True)

    normalize = subparsers.add_parser("normalize_docs")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)

    build = subparsers.add_parser("build_index")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)

    run = subparsers.add_parser("run_batch")
    run.add_argument("--questions", required=True)
    run.add_argument("--index", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--top-k", type=int, default=5)

    validate = subparsers.add_parser("validate_submission")
    validate.add_argument("--input", required=True)
    validate.add_argument("--questions")
    validate.add_argument("--max-issues", type=int, default=200)

    package = subparsers.add_parser("package_submission")
    package.add_argument("--input", required=True)
    package.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "ingest_corpus":
        return _cmd_ingest(args.input, args.output)
    if args.command == "normalize_docs":
        return _cmd_normalize_docs(args.input, args.output)
    if args.command == "build_index":
        return _cmd_build_index(args.input, args.output)
    if args.command == "run_batch":
        return _cmd_run_batch(args.questions, args.index, args.output, args.top_k)
    if args.command == "validate_submission":
        return _cmd_validate(args.input, args.questions, args.max_issues)
    if args.command == "package_submission":
        return _cmd_package(args.input, args.output)
    raise AssertionError(args.command)


def _cmd_ingest(input_path: str, output_path: str) -> int:
    articles, warnings = ingest_corpus(input_path)
    write_articles_jsonl(articles, output_path)
    _write_report(Path(output_path).with_suffix(".report.json"), {"articles": len(articles), "warnings": warnings})
    print(f"wrote {len(articles)} articles to {output_path}")
    if warnings:
        print(f"warnings: {len(warnings)}")
    return 0


def _cmd_normalize_docs(input_path: str, output_path: str) -> int:
    result = normalize_documents(input_path, output_path)
    print(f"normalized documents: {len(result.documents)}")
    print(f"manifest: {Path(output_path) / 'manifest.json'}")
    print(f"documents jsonl: {Path(output_path) / 'documents.jsonl'}")
    if result.report.get("failures"):
        print(f"failures: {len(result.report['failures'])}")
    if result.report.get("legacy_doc_files"):
        print(f"legacy .doc files: {result.report['legacy_doc_files']}")
    if result.report.get("duplicate_doc_ids"):
        print(f"duplicate doc_ids: {', '.join(result.report['duplicate_doc_ids'])}")
    return 1 if result.report.get("failures") else 0


def _cmd_build_index(input_path: str, output_path: str) -> int:
    articles = read_articles_jsonl(input_path)
    index = BM25Index(articles)
    index.save(output_path)
    print(f"indexed {len(articles)} articles to {output_path}")
    return 0


def _cmd_run_batch(questions_path: str, index_path: str, output_path: str, top_k: int) -> int:
    questions = load_questions(questions_path)
    index = BM25Index.load(index_path)
    predictions = []
    verifier_issues: list[dict] = []
    for question in questions:
        articles = retrieve_articles(index, question.question, top_k=top_k)
        answer = generate_grounded_answer(question.question, articles, max_articles=top_k)
        verification = verify_prediction_evidence(answer, articles)
        if not verification.ok:
            verifier_issues.append({"id": question.id, "issues": verification.issues})
        predictions.append(format_prediction(question, answer, articles))
    write_predictions(predictions, output_path)
    _write_report(
        Path(output_path).with_suffix(".manifest.json"),
        {"questions": len(questions), "index": index_path, "top_k": top_k, "verifier_issues": verifier_issues[:200]},
    )
    print(f"wrote {len(predictions)} predictions to {output_path}")
    if verifier_issues:
        print(f"verifier issues: {len(verifier_issues)}")
    return 0


def _cmd_validate(input_path: str, questions_path: str | None, max_issues: int) -> int:
    issues = validate_submission(input_path, questions_path)
    for issue in issues[:max_issues]:
        print(issue)
    if len(issues) > max_issues:
        print(f"... truncated {len(issues) - max_issues} more issues")
    print(f"validation issues: {len(issues)}")
    return 1 if issues else 0


def _cmd_package(input_path: str, output_path: str) -> int:
    issues = validate_submission(input_path)
    if issues:
        for issue in issues:
            print(issue)
        print("refusing to package invalid submission")
        return 1
    package_submission(input_path, output_path)
    print(f"wrote {output_path}")
    return 0


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
