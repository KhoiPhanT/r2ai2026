from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from legal_rag.corpus.ingest import ingest_corpus, read_articles_jsonl, write_articles_jsonl
from legal_rag.corpus.vbpl import (
    VbplImportConfig,
    clean_generated_data,
    import_vbpl_corpus,
    inspect_vbpl_corpus,
)
from legal_rag.documents import normalize_documents
from legal_rag.domain.components import component_matches
from legal_rag.evaluation import (
    build_prediction_map,
    build_trace_map,
    build_gold_scaffolds,
    compute_metadata_metrics,
    compute_prediction_metrics,
    load_gold_annotations,
    write_jsonl_rows,
)
from legal_rag.formatting.submission import (
    format_prediction,
    load_questions,
    package_submission,
    validate_package_manifest,
    validate_submission,
    write_predictions,
)
from legal_rag.generation import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    EvidenceAnswer,
    articles_from_used_evidence,
    answer_runtime_config,
    build_evidence_blocks,
    finalize_evidence_answer,
    generate_evidence_answer,
    repair_evidence_answer,
    OllamaConfig,
    OllamaError,
    generate_grounded_answer,
)
from legal_rag.lexicon import (
    LegalLexiconEntry,
    LegalLexiconIndex,
    build_legal_lexicon,
    load_legal_lexicon,
    normalize_legal_query_text,
    search_legal_lexicon,
)
from legal_rag.planner import (
    LegalQueryPlan,
    PlannedQuery,
    fallback_plan_for_debug,
    plan_legal_query,
    planner_runtime_config,
)
from legal_rag.question_metadata import infer_runtime_metadata, split_questions_for_gold
from legal_rag.retrieval import (
    BM25Index,
    FTS5Index,
    HybridRetrievalConfig,
    HybridRetrievalError,
    HybridRetriever,
    build_hybrid_index,
    build_fts5_index,
    config_from_mapping,
    evaluate_retrieval,
    retrieve_articles,
)
from legal_rag.schemas.models import ArticleNode, PredictedQuestionMetadata, Question, QuestionRunTrace
from legal_rag.utils.text import normalize_for_match, tokenize
from legal_rag.verifier import VerificationResult, build_evidence_metadata, verify_prediction_evidence
from legal_rag.verifier import verify_used_evidence_answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="legal_rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest_corpus")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--output", required=True)

    inspect_vbpl = subparsers.add_parser("inspect_vbpl_corpus")
    inspect_vbpl.add_argument("--input", default="data/law_data_raw")
    inspect_vbpl.add_argument("--report", default="data/normalized/vbpl_inspect_report.json")

    import_vbpl = subparsers.add_parser("import_vbpl_corpus")
    import_vbpl.add_argument("--input", default="data/law_data_raw")
    import_vbpl.add_argument("--output", default="data/normalized")
    import_vbpl.add_argument("--include-decisions", action="store_true")
    import_vbpl.add_argument("--include-english-like", action="store_true")

    clean_data = subparsers.add_parser("clean_generated_data")
    clean_data.add_argument("--data-root", default="data")

    normalize = subparsers.add_parser("normalize_docs")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)

    build = subparsers.add_parser("build_index")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)

    lexical_build = subparsers.add_parser("build_lexical_index")
    lexical_build.add_argument("--input", default="data/normalized/articles.jsonl")
    lexical_build.add_argument("--output", default="data/indices/articles_fts.sqlite")

    hybrid_build = subparsers.add_parser("build_hybrid_index")
    hybrid_build.add_argument("--input", required=True)
    hybrid_build.add_argument("--config", default="configs/local_m4.json")
    hybrid_build.add_argument("--report", default="data/indices/hybrid_index_report.json")

    lexicon_build = subparsers.add_parser("build_legal_lexicon")
    lexicon_build.add_argument("--input", default="data/normalized/articles.jsonl")
    lexicon_build.add_argument("--output", default="data/indices/legal_lexicon.jsonl")
    lexicon_build.add_argument("--max-entries", type=int, default=80000)

    plan_query = subparsers.add_parser("plan_query")
    plan_query.add_argument("--question", required=True)
    plan_query.add_argument("--config")
    plan_query.add_argument("--model")
    plan_query.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    plan_query.add_argument("--max-tokens", type=int, default=900)

    prepare_gold = subparsers.add_parser("prepare_gold_metadata")
    prepare_gold.add_argument("--questions", default="data/test.json")
    prepare_gold.add_argument("--output-dir", default="data/questions")
    prepare_gold.add_argument("--config")
    prepare_gold.add_argument("--model")
    prepare_gold.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    prepare_gold.add_argument("--max-tokens", type=int, default=900)

    run = subparsers.add_parser("run_batch")
    run.add_argument("--questions", required=True)
    run.add_argument("--index", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--backend")
    run.add_argument("--articles")
    run.add_argument("--config")
    run.add_argument("--allow-verifier-issues", action="store_true")
    run.add_argument("--failure-policy", choices=["stop", "quarantine"], default="stop")
    run.add_argument("--resume-from")
    run.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    run.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    run.add_argument("--max-tokens", type=int, default=700)

    ask = subparsers.add_parser("ask")
    ask.add_argument("--question", required=True)
    ask.add_argument("--index", required=True)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--backend")
    ask.add_argument("--articles")
    ask.add_argument("--config")
    ask.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    ask.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ask.add_argument("--max-tokens", type=int, default=700)

    debug = subparsers.add_parser("debug_retrieval")
    debug.add_argument("--index", required=True)
    debug.add_argument("--question")
    debug.add_argument("--questions")
    debug.add_argument("--output")
    debug.add_argument("--backend")
    debug.add_argument("--articles")
    debug.add_argument("--config")
    debug.add_argument("--top-k", type=int, default=5)

    debug_pipeline = subparsers.add_parser("debug_pipeline")
    debug_pipeline.add_argument("--question", required=True)
    debug_pipeline.add_argument("--index")
    debug_pipeline.add_argument("--top-k", type=int, default=8)
    debug_pipeline.add_argument("--backend")
    debug_pipeline.add_argument("--articles")
    debug_pipeline.add_argument("--config", default="configs/local_m4.json")
    debug_pipeline.add_argument("--model")
    debug_pipeline.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    debug_pipeline.add_argument("--max-tokens", type=int, default=900)

    eval_cmd = subparsers.add_parser("eval_retrieval")
    eval_cmd.add_argument("--questions", required=True)
    eval_cmd.add_argument("--expected", required=True)
    eval_cmd.add_argument("--index", default="data/indices/bm25_index.json")
    eval_cmd.add_argument("--articles")
    eval_cmd.add_argument("--backend")
    eval_cmd.add_argument("--config", default="configs/local_m4.json")
    eval_cmd.add_argument("--top-k", type=int, default=5)

    eval_pipeline = subparsers.add_parser("eval_pipeline")
    eval_pipeline.add_argument("--questions", required=True)
    eval_pipeline.add_argument("--expected", required=True)
    eval_pipeline.add_argument("--index", default="data/indices/bm25_index.json")
    eval_pipeline.add_argument("--articles")
    eval_pipeline.add_argument("--config", default="configs/local_m4.json")
    eval_pipeline.add_argument("--top-k", type=int, default=5)
    eval_pipeline.add_argument("--model")
    eval_pipeline.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    eval_pipeline.add_argument("--max-tokens", type=int, default=900)

    validate = subparsers.add_parser("validate_submission")
    validate.add_argument("--input", required=True)
    validate.add_argument("--questions")
    validate.add_argument("--max-issues", type=int, default=200)

    package = subparsers.add_parser("package_submission")
    package.add_argument("--input", required=True)
    package.add_argument("--output", required=True)
    package.add_argument("--allow-verifier-issues", action="store_true")

    submit = subparsers.add_parser("submit")
    submit.add_argument("--questions", default="data/test.json")
    submit.add_argument("--index")
    submit.add_argument("--output-dir", default="output")
    submit.add_argument("--top-k", type=int, default=5)
    submit.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    submit.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    submit.add_argument("--max-tokens", type=int, default=700)
    submit.add_argument("--config")
    submit.add_argument("--backend")
    submit.add_argument("--articles")
    submit.add_argument("--allow-verifier-issues", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "ingest_corpus":
        return _cmd_ingest(args.input, args.output)
    if args.command == "inspect_vbpl_corpus":
        return _cmd_inspect_vbpl_corpus(args.input, args.report)
    if args.command == "import_vbpl_corpus":
        return _cmd_import_vbpl_corpus(
            args.input,
            args.output,
            include_decisions=args.include_decisions,
            include_english_like=args.include_english_like,
        )
    if args.command == "clean_generated_data":
        return _cmd_clean_generated_data(args.data_root)
    if args.command == "normalize_docs":
        return _cmd_normalize_docs(args.input, args.output)
    if args.command == "build_index":
        return _cmd_build_index(args.input, args.output)
    if args.command == "build_lexical_index":
        return _cmd_build_lexical_index(args.input, args.output)
    if args.command == "build_hybrid_index":
        return _cmd_build_hybrid_index(args.input, args.config, args.report)
    if args.command == "build_legal_lexicon":
        return _cmd_build_legal_lexicon(args.input, args.output, args.max_entries)
    if args.command == "plan_query":
        config = _load_config(args.config)
        planner_config = _planner_config(config, args.model, args.ollama_url, args.max_tokens)
        return _cmd_plan_query(args.question, planner_config, _load_lexicon_from_config(config))
    if args.command == "prepare_gold_metadata":
        config = _load_config(args.config)
        planner_config = _planner_config(config, args.model, args.ollama_url, args.max_tokens)
        return _cmd_prepare_gold_metadata(args.questions, args.output_dir, planner_config, _load_lexicon_from_config(config))
    if args.command == "run_batch":
        config = _load_config(args.config)
        return _cmd_run_batch(
            args.questions,
            _resolve_index_path(args.index, config),
            args.output,
            args.top_k,
            _answer_config(config, args.model, args.ollama_url, args.max_tokens),
            backend=_resolve_backend(args.backend, config),
            articles_path=_resolve_articles_path(args.articles, config),
            config=config,
            allow_verifier_issues=args.allow_verifier_issues,
            failure_policy=args.failure_policy,
            resume_from=args.resume_from,
        )
    if args.command == "ask":
        config = _load_config(args.config)
        return _cmd_ask(
            args.question,
            _resolve_index_path(args.index, config),
            args.top_k,
            _answer_config(config, args.model, args.ollama_url, args.max_tokens),
            backend=_resolve_backend(args.backend, config),
            articles_path=_resolve_articles_path(args.articles, config),
            config=config,
        )
    if args.command == "debug_retrieval":
        config = _load_config(args.config)
        return _cmd_debug_retrieval(
            _resolve_index_path(args.index, config),
            args.question,
            args.questions,
            args.output,
            args.top_k,
            backend=_resolve_backend(args.backend, config),
            articles_path=_resolve_articles_path(args.articles, config),
            config=config,
        )
    if args.command == "debug_pipeline":
        config = _load_config(args.config)
        model = args.model or config.get("generation", {}).get("target_model", DEFAULT_OLLAMA_MODEL)
        return _cmd_debug_pipeline(
            args.question,
            _resolve_index_path(args.index, config),
            args.top_k,
            _answer_config(config, model, args.ollama_url, args.max_tokens),
            backend=_resolve_backend(args.backend, config),
            articles_path=_resolve_articles_path(args.articles, config),
            config=config,
        )
    if args.command == "eval_retrieval":
        config = _load_config(args.config)
        return _cmd_eval_retrieval(
            args.questions,
            args.expected,
            _resolve_index_path(args.index, config),
            _resolve_articles_path(args.articles, config),
            _resolve_backend(args.backend, config),
            config,
            args.top_k,
        )
    if args.command == "eval_pipeline":
        config = _load_config(args.config)
        return _cmd_eval_pipeline(
            args.questions,
            args.expected,
            _resolve_index_path(args.index, config),
            _resolve_articles_path(args.articles, config),
            config,
            args.top_k,
            _planner_config(config, args.model, args.ollama_url, args.max_tokens),
        )
    if args.command == "validate_submission":
        return _cmd_validate(args.input, args.questions, args.max_issues)
    if args.command == "package_submission":
        return _cmd_package(args.input, args.output, args.allow_verifier_issues)
    if args.command == "submit":
        return _cmd_submit(args)
    raise AssertionError(args.command)


def _cmd_ingest(input_path: str, output_path: str) -> int:
    articles, warnings = ingest_corpus(input_path)
    write_articles_jsonl(articles, output_path)
    _write_report(Path(output_path).with_suffix(".report.json"), {"articles": len(articles), "warnings": warnings})
    print(f"wrote {len(articles)} articles to {output_path}")
    if warnings:
        print(f"warnings: {len(warnings)}")
    return 0


def _cmd_inspect_vbpl_corpus(input_path: str, report_path: str) -> int:
    report = inspect_vbpl_corpus(input_path, report_path)
    documents = report.documents
    units = report.legal_units
    print(f"documents: {documents.get('valid_records', 0)} valid records")
    print(f"legal units: {units.get('valid_records', 0)} valid records")
    print(f"document parse errors: {documents.get('parse_error_count', 0)}")
    print(f"legal unit parse errors: {units.get('parse_error_count', 0)}")
    print(f"report: {report_path}")
    return 0 if not documents.get("parse_error_count") and not units.get("parse_error_count") else 1


def _cmd_import_vbpl_corpus(
    input_path: str,
    output_path: str,
    *,
    include_decisions: bool,
    include_english_like: bool,
) -> int:
    report = import_vbpl_corpus(
        input_path,
        output_path,
        VbplImportConfig(include_decisions=include_decisions, include_english_like=include_english_like),
    )
    print(f"selected documents: {report.selected_documents}")
    print(f"wrote documents: {report.written_documents}")
    print(f"wrote articles: {report.written_articles}")
    print(f"wrote legal units: {report.written_legal_units}")
    print(f"fallback articles: {report.fallback_articles}")
    print(f"report: {Path(output_path) / 'vbpl_import_report.json'}")
    return 0 if not report.parse_errors else 1


def _cmd_clean_generated_data(data_root: str) -> int:
    try:
        removed = clean_generated_data(data_root)
    except ValueError as exc:
        print(str(exc))
        return 1
    if removed:
        print("removed generated data:")
        for path in removed:
            print(f"- {path}")
    else:
        print("no generated data directories to remove")
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


def _cmd_build_lexical_index(input_path: str, output_path: str) -> int:
    report = build_fts5_index(input_path, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_build_hybrid_index(input_path: str, config_path: str, report_path: str) -> int:
    config = _load_config(config_path)
    try:
        report = build_hybrid_index(
            input_path,
            config_from_mapping(config),
            progress=lambda message: print(f"[build-hybrid] {message}", file=sys.stderr, flush=True),
        )
    except HybridRetrievalError as exc:
        print(f"hybrid index build failed: {exc}")
        return 1
    _write_report(Path(report_path), report.to_dict())
    print(f"indexed {report.articles} articles to qdrant collection {report.collection}")
    print(f"report: {report_path}")
    return 0


def _cmd_build_legal_lexicon(input_path: str, output_path: str, max_entries: int) -> int:
    report = build_legal_lexicon(input_path, output_path, max_entries=max_entries)
    print(f"lexicon entries: {report['entries']}")
    print(f"articles scanned: {report['articles']}")
    print(f"output: {report['output']}")
    return 0


def _cmd_plan_query(question: str, planner_config: OllamaConfig, lexicon: list[LegalLexiconEntry] | None = None) -> int:
    try:
        plan = plan_legal_query(question, planner_config, lexicon_candidates=_lexicon_matches(lexicon, question))
    except OllamaError as exc:
        print(f"planner failed: {exc}")
        return 1
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_prepare_gold_metadata(
    questions_path: str,
    output_dir: str,
    planner_config: OllamaConfig,
    lexicon: list[LegalLexiconEntry] | None = None,
) -> int:
    questions = load_questions(questions_path)
    predicted_by_id: dict[int, PredictedQuestionMetadata] = {}
    planner_failures: list[int] = []
    for question in questions:
        try:
            plan = plan_legal_query(question.question, planner_config, lexicon_candidates=_lexicon_matches(lexicon, question.question))
        except OllamaError:
            planner_failures.append(question.id)
            plan = fallback_plan_for_debug(question.question, lexicon_candidates=_lexicon_matches(lexicon, question.question))
        predicted_by_id[question.id] = plan.predicted_metadata()

    tune_questions, holdout_questions = split_questions_for_gold(questions, predicted_by_id=predicted_by_id)
    tune_rows, holdout_rows = build_gold_scaffolds(questions, tune_questions, holdout_questions, predicted_by_id)
    output_root = Path(output_dir)
    tune_path = output_root / "dev_gold.jsonl"
    holdout_path = output_root / "holdout_gold.jsonl"
    write_jsonl_rows(tune_path, tune_rows)
    write_jsonl_rows(holdout_path, holdout_rows)
    _write_report(
        output_root / "gold_split_manifest.json",
        {
            "questions": len(questions),
            "tune": len(tune_rows),
            "holdout": len(holdout_rows),
            "planner_failure_count": len(planner_failures),
            "planner_failure_preview": planner_failures[:20],
            "planner_model": planner_config.model,
            "ollama_url": planner_config.url,
        },
    )
    print(f"wrote {tune_path}")
    print(f"wrote {holdout_path}")
    if planner_failures:
        preview = planner_failures[:20]
        suffix = "..." if len(planner_failures) > len(preview) else ""
        print(f"warning: fallback planner used for {len(planner_failures)} questions: {preview}{suffix}")
    return 0


def _cmd_run_batch(
    questions_path: str,
    index_path: str,
    output_path: str,
    top_k: int,
    ollama_config: OllamaConfig,
    *,
    backend: str,
    articles_path: str,
    config: dict,
    allow_verifier_issues: bool = False,
    failure_policy: str = "stop",
    resume_from: str | None = None,
) -> int:
    questions = load_questions(questions_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out.with_suffix(".progress.jsonl")
    run_state_path = out.parent / f".{out.stem}.run.json"
    run_state = _batch_run_state(
        questions_path=questions_path,
        index_path=index_path,
        articles_path=articles_path,
        backend=backend,
        top_k=top_k,
        model=ollama_config.model,
        ollama_url=ollama_config.url,
        config=config,
        resume_from=resume_from,
    )
    if resume_from and not progress_path.exists() and not run_state_path.exists():
        _write_report(run_state_path, run_state)
    try:
        searcher, retrieval_manifest = _load_retrieval_backend(backend, index_path, articles_path, config)
    except HybridRetrievalError as exc:
        print(f"retrieval backend failed: {exc}")
        return 1
    if resume_from and not progress_path.exists():
        seed_path = Path(resume_from)
        if not seed_path.exists():
            print(f"resume seed not found: {seed_path}")
            return 1
        valid_ids = {question.id for question in questions}
        seeded: list[dict] = []
        rejected: list[dict] = []
        seen_seed_ids: set[int] = set()
        questions_by_id = {question.id: question for question in questions}
        for line_number, line in enumerate(seed_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_id = int(record["id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                print(f"resume seed invalid at line {line_number}: {seed_path}")
                return 1
            if record_id not in valid_ids or record_id in seen_seed_ids:
                continue
            seen_seed_ids.add(record_id)
            issues = _resume_seed_issues(record, questions_by_id[record_id], searcher)
            if issues:
                rejected.append({"id": record_id, "question": questions_by_id[record_id].question, "issues": issues})
                continue
            seeded.append(record)
        with progress_path.open("w", encoding="utf-8") as seed_file:
            for record in seeded:
                seed_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"seeded {len(seeded)} completed predictions from {seed_path}")
        if rejected:
            rejected_path = out.with_suffix(".resume_rejected.jsonl")
            with rejected_path.open("w", encoding="utf-8") as rejected_file:
                for record in rejected:
                    rejected_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"resume revalidation rejected {len(rejected)} records -> {rejected_path}")
    if progress_path.exists():
        if not run_state_path.exists():
            if resume_from:
                _write_report(run_state_path, run_state)
            else:
                print(f"resume refused: missing run fingerprint {run_state_path}")
                print("use a new --output path, or remove the old progress file after reviewing it")
                return 1
        try:
            previous_run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"resume refused: invalid run fingerprint {run_state_path}")
            return 1
        if previous_run_state.get("fingerprint") != run_state["fingerprint"]:
            print("resume refused: run fingerprint changed")
            print(f"previous={previous_run_state.get('fingerprint', '')} current={run_state['fingerprint']}")
            print("use a new --output path so predictions from different configs are not mixed")
            return 1
    if not run_state_path.exists():
        _write_report(run_state_path, run_state)
    failed_path = out.with_suffix(".failed.jsonl")
    failed_trace_path = out.with_suffix(".failed.trace.jsonl")
    unresolved_failures = _load_jsonl_by_id(failed_path)
    unresolved_failure_traces = _load_jsonl_by_id(failed_trace_path)

    # --- Resume: load already-completed predictions ---
    done_by_id: dict[int, dict] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    record = json.loads(line)
                    done_by_id[int(record["id"])] = record
                except (json.JSONDecodeError, KeyError):
                    pass
    initial_done = len(done_by_id)
    if initial_done > 0:
        print(f"resuming: {initial_done}/{len(questions)} already completed")
    remaining = [q for q in questions if q.id not in done_by_id]
    total = len(questions)
    verifier_issues: list[dict] = []
    trace_rows: list[QuestionRunTrace] = []
    failed_rows: list[dict] = []
    batch_start = time.time()
    planner_config = _planner_config(config, ollama_config.model, ollama_config.url, 900)
    lexicon = _load_lexicon_from_config(config)
    if initial_done:
        print(f"progress: {progress_path}")
    if failure_policy == "quarantine":
        print(f"failure policy: quarantine -> {failed_path}")

    # --- Stream: process remaining questions one by one ---
    with progress_path.open("a", encoding="utf-8") as progress_file, failed_path.open("a", encoding="utf-8") as failed_file, failed_trace_path.open("a", encoding="utf-8") as failed_trace_file:
        for step, question in enumerate(remaining, start=1):
            q_start = time.time()
            done_count = initial_done + step
            try:
                pred, trace, verification = _answer_question(
                    question,
                    searcher,
                    top_k,
                    planner_config,
                    ollama_config,
                    max_context_articles=_max_context_articles(config),
                    lexicon=lexicon,
                )
            except (OllamaError, HybridRetrievalError) as exc:
                elapsed = time.time() - batch_start
                if failure_policy == "quarantine":
                    failed = {"id": question.id, "question": question.question, "error": str(exc), "stage": "exception"}
                    failed_file.write(json.dumps(failed, ensure_ascii=False) + "\n")
                    failed_file.flush()
                    unresolved_failures[question.id] = failed
                    failed_rows.append(failed)
                    print(f"[{done_count}/{total}] id={question.id} ⚠ quarantined: {exc} ({elapsed:.0f}s elapsed)")
                    continue
                print(f"\n[{done_count}/{total}] FAIL id={question.id}: {exc} ({elapsed:.0f}s elapsed)")
                print("run_batch is fail-closed; no final results.json will be written.")
                return 1

            if not verification.ok:
                verifier_issues.append({"id": question.id, "issues": verification.issues})
                if not allow_verifier_issues:
                    if failure_policy == "quarantine":
                        failed = {
                            "id": question.id,
                            "question": question.question,
                            "issues": verification.issues,
                            "answer": getattr(pred, "answer", ""),
                            "relevant_docs": getattr(pred, "relevant_docs", []),
                            "relevant_articles": getattr(pred, "relevant_articles", []),
                            "stage": "verifier",
                        }
                        failed_file.write(json.dumps(failed, ensure_ascii=False) + "\n")
                        failed_file.flush()
                        failed_trace_file.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
                        failed_trace_file.flush()
                        unresolved_failures[question.id] = failed
                        unresolved_failure_traces[question.id] = trace.to_dict()
                        failed_rows.append(failed)
                        q_elapsed = time.time() - q_start
                        total_elapsed = time.time() - batch_start
                        print(
                            f"[{done_count}/{total}] id={question.id} ⚠ quarantined"
                            f" issues={verification.issues} {q_elapsed:.1f}s (total {total_elapsed:.0f}s)"
                        )
                        continue
                    print(f"\n[{done_count}/{total}] verifier failed id={question.id}: {verification.issues}")
                    print("run_batch is fail-closed; no final results.json will be written.")
                    return 1

            pred_dict = pred.to_dict()
            trace_rows.append(trace)

            # Append to progress file immediately
            progress_file.write(json.dumps(pred_dict, ensure_ascii=False) + "\n")
            progress_file.flush()

            done_by_id[question.id] = pred_dict
            unresolved_failures.pop(question.id, None)
            unresolved_failure_traces.pop(question.id, None)

            # Streaming log
            q_elapsed = time.time() - q_start
            total_elapsed = time.time() - batch_start
            v_flag = " ⚠" if not verification.ok else " ✓"
            n_arts = len(trace.used_evidence_ids)
            print(
                f"[{done_count}/{total}] id={question.id}{v_flag}"
                f"  arts={n_arts}  {q_elapsed:.1f}s"
                f"  (total {total_elapsed:.0f}s)"
            )

    # --- Assemble final output from all predictions (done + newly done) ---
    # Maintain original question order
    all_predictions = []
    for q in questions:
        record = done_by_id.get(q.id)
        if record is not None:
            all_predictions.append(record)

    unresolved_failures = {
        record_id: record for record_id, record in unresolved_failures.items() if record_id not in done_by_id
    }
    unresolved_failure_traces = {
        record_id: record for record_id, record in unresolved_failure_traces.items() if record_id in unresolved_failures
    }
    _write_jsonl_by_id(failed_path, unresolved_failures)
    _write_jsonl_by_id(failed_trace_path, unresolved_failure_traces)
    incomplete_count = total - len(all_predictions)
    final_path = out if not unresolved_failures and incomplete_count == 0 else out.with_suffix(".partial.json")
    final_path.write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_report(
        out.with_suffix(".manifest.json"),
        {
            "questions": len(questions),
            "completed": len(all_predictions),
            "failed": len(unresolved_failures),
            "failure_policy": failure_policy,
            "failed_path": str(failed_path) if unresolved_failures else "",
            "index": index_path,
            "top_k": top_k,
            "planner_backend": "ollama",
            "planner_model": planner_config.model,
            **retrieval_manifest,
            "generator_backend": "ollama",
            "model": ollama_config.model,
            "ollama_url": ollama_config.url,
            "verifier_issues": verifier_issues[:200],
            "allow_verifier_issues": allow_verifier_issues,
        },
    )
    if trace_rows:
        trace_path = out.with_suffix(".trace.jsonl")
        with trace_path.open("w", encoding="utf-8") as trace_file:
            for row in trace_rows:
                trace_file.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    elapsed = time.time() - batch_start
    print(f"\nwrote {len(all_predictions)} predictions to {final_path} ({elapsed:.0f}s)")
    if unresolved_failures:
        print(f"quarantined: {len(unresolved_failures)} unresolved questions (see {failed_path})")
        return 1
    if verifier_issues:
        print(f"warning: {len(verifier_issues)} verifier issues (see manifest)")
    if incomplete_count:
        print(f"error: {incomplete_count} questions have no prediction")
        return 1
    return 0


def _batch_run_state(
    *,
    questions_path: str,
    index_path: str,
    articles_path: str,
    backend: str,
    top_k: int,
    model: str,
    ollama_url: str,
    config: dict,
    resume_from: str | None = None,
) -> dict:
    payload = {
        "schema_version": 1,
        "questions": _file_identity(questions_path, include_content_hash=True),
        "index": _file_identity(index_path),
        "articles": _file_identity(articles_path),
        "backend": backend,
        "top_k": int(top_k),
        "model": model,
        "ollama_url": ollama_url,
        "config": config,
        "runtime_artifacts": {
            "lexical_index": _file_identity(str(config.get("retrieval", {}).get("lexical_index") or "")),
            "lexicon": _file_identity(str(config.get("lexicon", {}).get("path") or "")),
        },
        "pipeline_code_sha256": _pipeline_code_digest(),
        "resume_from": _file_identity(resume_from, include_content_hash=True) if resume_from else None,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**payload, "fingerprint": hashlib.sha256(serialized.encode("utf-8")).hexdigest()}


def _pipeline_code_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_identity(path_value: str, *, include_content_hash: bool = False) -> dict:
    if not path_value:
        return {"path": "", "exists": False}
    path = Path(path_value)
    identity = {"path": str(path.resolve())}
    if not path.exists():
        return {**identity, "exists": False}
    stat = path.stat()
    identity.update({"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if include_content_hash:
        identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def _load_jsonl_by_id(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    output: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            output[int(record["id"])] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return output


def _write_jsonl_by_id(path: Path, records: dict[int, dict]) -> None:
    if not records:
        path.unlink(missing_ok=True)
        return
    with path.open("w", encoding="utf-8") as output:
        for record_id in sorted(records):
            output.write(json.dumps(records[record_id], ensure_ascii=False) + "\n")


def _resume_seed_issues(record: dict, question: Question, searcher: object) -> list[str]:
    if not isinstance(searcher, HybridRetriever) or not hasattr(searcher.lexical_index, "get_article"):
        return []
    article_keys = [str(item) for item in record.get("relevant_articles", []) if str(item).strip()]
    articles = [searcher.lexical_index.get_article(key) for key in article_keys]
    if not article_keys or any(article is None for article in articles):
        return ["resume_evidence_missing"]
    resolved = [article for article in articles if article is not None]
    metadata = infer_runtime_metadata(question.question)
    verification = verify_used_evidence_answer(
        str(record.get("answer") or ""),
        resolved,
        resolved,
        question=question.question,
        required_components=metadata.requested_components,
        target_norm_roles=metadata.target_norm_roles,
    )
    issues = [*verification.hard_issues, *verification.repairable_issues]
    issues.extend(warning for warning in verification.warnings if warning.startswith("semantic_"))
    if not _question_has_local_scope(question.question) and any(_record_article_is_local(article) for article in resolved):
        issues.append("resume_unrequested_local_scope")
    if hasattr(searcher.lexical_index, "superseding_sources"):
        superseding = searcher.lexical_index.superseding_sources({article.doc_id for article in resolved})
        for target, sources in sorted(superseding.items()):
            issues.append(f"resume_superseded_document:{target}<-{'|'.join(sorted(sources))}")
    return list(dict.fromkeys(issues))


def _question_has_local_scope(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("địa bàn", "tỉnh", "thành phố", "hđnd", "ubnd", "thủ đô"))


def _record_article_is_local(article: ArticleNode) -> bool:
    title = article.title_for_submission.lower()
    return article.doc_type.lower() == "nghị quyết" and any(
        term in title for term in ("hội đồng nhân dân", "hđnd", "địa bàn tỉnh", "địa bàn thành phố")
    )


def _cmd_ask(
    question: str,
    index_path: str,
    top_k: int,
    ollama_config: OllamaConfig,
    *,
    backend: str,
    articles_path: str,
    config: dict,
) -> int:
    try:
        searcher, _manifest = _load_retrieval_backend(backend, index_path, articles_path, config)
    except HybridRetrievalError as exc:
        print(f"retrieval backend failed: {exc}")
        return 1
    planner_config = _planner_config(config, ollama_config.model, ollama_config.url, 900)
    try:
        pred, trace, verification = _answer_question(
            Question(id=1, question=question),
            searcher,
            top_k,
            planner_config,
            ollama_config,
            max_context_articles=_max_context_articles(config),
            lexicon=_load_lexicon_from_config(config),
        )
    except (OllamaError, HybridRetrievalError) as exc:
        print(f"model generation failed: {exc}")
        return 1

    print(pred.answer)
    print("\nPredicted metadata:")
    print(json.dumps(trace.predicted_metadata.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nBackend thực tế: {trace.actual_backend or _backend_name(searcher)}")
    print(
        f"Stage timing: planner={trace.planner_ms:.0f}ms retrieval={trace.retrieval_ms:.0f}ms "
        f"rerank={trace.rerank_ms:.0f}ms answer={trace.answer_ms:.0f}ms"
    )
    print("\nPlanned queries:")
    for query in trace.predicted_metadata.planned_queries:
        print(f"- {query}")
    print("\nCăn cứ đã dùng:")
    for article in pred.relevant_articles:
        print(f"- {article}")
    if not verification.ok:
        print(f"\nverifier issues: {verification.issues}")
        return 1
    return 0


def _cmd_debug_pipeline(
    question: str,
    index_path: str,
    top_k: int,
    ollama_config: OllamaConfig,
    *,
    backend: str,
    articles_path: str,
    config: dict,
) -> int:
    try:
        searcher, retrieval_manifest = _load_retrieval_backend(backend, index_path, articles_path, config)
        lexicon = _load_lexicon_from_config(config)
        corpus_candidates = _resolve_corpus_candidates(searcher, question)
        plan = plan_legal_query(
            question,
            _planner_config(config, ollama_config.model, ollama_config.url, ollama_config.max_tokens),
            lexicon_candidates=_lexicon_matches(lexicon, question),
            corpus_candidates=corpus_candidates,
        )
        articles = _search_articles(searcher, question, top_k, plan=plan)
        retry_articles, _retry_reason = _retry_retrieval_if_drift(searcher, question, top_k, plan, articles)
        if retry_articles is not None:
            articles = retry_articles
    except (HybridRetrievalError, OllamaError) as exc:
        print(f"debug pipeline failed: {exc}")
        return 1
    max_context = _max_context_articles(config)
    blocks = build_evidence_blocks(articles[:max_context] if max_context else articles)
    print(
        json.dumps(
            {
                "retrieval": retrieval_manifest,
                "plan": plan.to_dict(),
                "predicted_metadata": plan.predicted_metadata().to_dict(),
                "corpus_candidates": [candidate.to_dict() for candidate in corpus_candidates],
                "actual_backend": _retrieval_trace(searcher).get("actual_backend", _backend_name(searcher)),
                "candidate_counts": _retrieval_trace(searcher).get("candidate_counts", {}),
                "retrieval_budget": _retrieval_trace(searcher).get("retrieval_budget", {}),
                "planned_query_count": _retrieval_trace(searcher).get("planned_query_count", 0),
                "semantic_query_count": _retrieval_trace(searcher).get("semantic_query_count", 0),
                "query_branches": _retrieval_trace(searcher).get("query_branches", []),
                "graph_expansions": _retrieval_trace(searcher).get("graph_expansions", []),
                "superseded_doc_ids": _retrieval_trace(searcher).get("superseded_doc_ids", []),
                "scope_filtered_doc_ids": _retrieval_trace(searcher).get("scope_filtered_doc_ids", []),
                "threshold_cutoff_reason": _retrieval_trace(searcher).get("threshold_cutoff_reason", ""),
                "qdrant_ms": _retrieval_trace(searcher).get("qdrant_ms", 0),
                "rerank_ms": _retrieval_trace(searcher).get("rerank_ms", 0),
                "max_context_articles": max_context,
                "evidence": [block.to_dict() for block in blocks],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_debug_retrieval(
    index_path: str,
    question: str | None,
    questions_path: str | None,
    output_path: str | None,
    top_k: int,
    *,
    backend: str,
    articles_path: str,
    config: dict,
) -> int:
    if bool(question) == bool(questions_path):
        print("provide exactly one of --question or --questions")
        return 1

    try:
        searcher, retrieval_manifest = _load_retrieval_backend(backend, index_path, articles_path, config)
    except HybridRetrievalError as exc:
        print(f"retrieval backend failed: {exc}")
        return 1
    questions = load_questions(questions_path) if questions_path else [Question(id=1, question=str(question))]
    predictions = []
    verifier_issues: list[dict] = []
    for item in questions:
        articles = _search_articles(searcher, item.question, top_k)
        answer = generate_grounded_answer(item.question, articles, max_articles=top_k)
        verification = verify_prediction_evidence(answer, articles)
        if not verification.ok:
            verifier_issues.append({"id": item.id, "issues": verification.issues})
        predictions.append(format_prediction(item, answer, articles))

    manifest = {
        "questions": len(questions),
        "index": index_path,
        "top_k": top_k,
        **retrieval_manifest,
        "generator_backend": "template_debug",
        "model": "",
        "ollama_url": "",
        "verifier_issues": verifier_issues[:200],
    }
    if output_path:
        write_predictions(predictions, output_path)
        _write_report(Path(output_path).with_suffix(".manifest.json"), manifest)
        print(f"wrote debug retrieval output to {output_path}")
    else:
        print(json.dumps({"manifest": manifest, "predictions": [p.to_dict() for p in predictions]}, ensure_ascii=False, indent=2))
    return 0


def _answer_question(
    question: Question,
    searcher: object,
    top_k: int,
    planner_config: OllamaConfig,
    answer_config: OllamaConfig,
    *,
    max_context_articles: int | None = None,
    lexicon: list[LegalLexiconEntry] | None = None,
) -> tuple[object, QuestionRunTrace, object]:
    planner_start = time.perf_counter()
    effective_planner_config = planner_runtime_config(question.question, planner_config)
    corpus_candidates = _resolve_corpus_candidates(searcher, question.question)
    plan = plan_legal_query(
        question.question,
        planner_config,
        lexicon_candidates=_lexicon_matches(lexicon, question.question),
        corpus_candidates=corpus_candidates,
    )
    planner_ms = (time.perf_counter() - planner_start) * 1000.0
    retrieval_start = time.perf_counter()
    retrieval_top_k = _adaptive_retrieval_top_k(plan, top_k)
    articles = _search_articles(searcher, question.question, retrieval_top_k, plan=plan)
    repair_attempts: list[dict] = []
    missing_before_answer = [
        component for component in plan.requested_components if not _context_supports_component(component, articles)
    ]
    if missing_before_answer:
        targeted_components = missing_before_answer[:3]
        repaired_articles = _targeted_components_retrieval(
            searcher,
            question.question,
            retrieval_top_k,
            plan,
            articles,
            targeted_components,
        )
        improved_components = [
            component for component in targeted_components if _context_supports_component(component, repaired_articles)
        ]
        if improved_components:
            articles = repaired_articles
            repair_attempts.append(
                {
                    "stage": "targeted_retrieval",
                    "components": targeted_components,
                    "improved_components": improved_components,
                    "query_count": len(targeted_components),
                }
            )
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
    answer_start = time.perf_counter()
    effective_answer_config = answer_runtime_config(plan, answer_config)
    context_limit = _adaptive_context_limit(plan, max_context_articles)
    context_articles = _select_evidence_candidates(plan, articles, context_limit)
    evidence_blocks = build_evidence_blocks(context_articles, max_chars=_adaptive_evidence_chars(plan))
    evidence_answer = _insufficient_answer_for_corpus_gap(plan, evidence_blocks)
    if evidence_answer is None:
        evidence_answer = generate_evidence_answer(question.question, plan, evidence_blocks, answer_config)
    else:
        repair_attempts.append(
            {
                "stage": "corpus_gap_guard",
                "components": evidence_answer.insufficient_components,
                "llm_calls": 0,
            }
        )
    answer_ms = (time.perf_counter() - answer_start) * 1000.0
    used_articles = articles_from_used_evidence(evidence_blocks, evidence_answer.used_evidence_ids)
    _repair_list_answer_from_used_evidence(question.question, evidence_answer, used_articles)
    used_articles = finalize_evidence_answer(evidence_answer, evidence_blocks)
    if evidence_answer.insufficient_evidence and not used_articles:
        verification = VerificationResult(ok=True, issues=[], warnings=["insufficient_evidence"])
    else:
        verification = verify_used_evidence_answer(
            evidence_answer.answer,
            used_articles,
            context_articles,
            question=question.question,
            required_components=plan.requested_components,
            target_norm_roles=plan.target_norm_roles,
            covered_components=evidence_answer.covered_components,
            insufficient_components=evidence_answer.insufficient_components,
            claims=evidence_answer.claims,
            evidence_by_id={block.evidence_id: block.article for block in evidence_blocks},
        )
    missing_components = _component_coverage_missing(verification.issues)
    auto_insufficient = [
        component for component in missing_components if not _context_supports_component(component, context_articles)
    ]
    if auto_insufficient:
        evidence_answer.answer = _append_insufficient_component_note(evidence_answer.answer, auto_insufficient)
        for component in auto_insufficient:
            if component not in evidence_answer.insufficient_components:
                evidence_answer.insufficient_components.append(component)
        verification = verify_used_evidence_answer(
            evidence_answer.answer,
            used_articles,
            context_articles,
            question=question.question,
            required_components=plan.requested_components,
            target_norm_roles=plan.target_norm_roles,
            covered_components=evidence_answer.covered_components,
            insufficient_components=evidence_answer.insufficient_components,
            claims=evidence_answer.claims,
            evidence_by_id={block.evidence_id: block.article for block in evidence_blocks},
        )
    repairable_missing = [
        component
        for component in _component_coverage_missing(list(verification.repairable_issues or []))
        if component not in auto_insufficient and _context_supports_component(component, context_articles)
    ]
    deterministic_components = _repair_missing_components_from_direct_evidence(
        evidence_answer,
        evidence_blocks,
        repairable_missing,
    )
    if deterministic_components:
        used_articles = finalize_evidence_answer(evidence_answer, evidence_blocks)
        repair_attempts.append(
            {"stage": "deterministic_component_repair", "components": deterministic_components, "llm_calls": 0}
        )
        verification = verify_used_evidence_answer(
            evidence_answer.answer,
            used_articles,
            context_articles,
            question=question.question,
            required_components=plan.requested_components,
            target_norm_roles=plan.target_norm_roles,
            covered_components=evidence_answer.covered_components,
            insufficient_components=evidence_answer.insufficient_components,
            claims=evidence_answer.claims,
            evidence_by_id={block.evidence_id: block.article for block in evidence_blocks},
        )
        repairable_missing = [
            component
            for component in _component_coverage_missing(list(verification.repairable_issues or []))
            if component not in auto_insufficient and _context_supports_component(component, context_articles)
        ]
    if repairable_missing:
        repair_start = time.perf_counter()
        evidence_answer = repair_evidence_answer(
            question.question,
            plan,
            evidence_blocks,
            evidence_answer,
            repairable_missing,
            answer_config,
        )
        used_articles = articles_from_used_evidence(evidence_blocks, evidence_answer.used_evidence_ids)
        _repair_list_answer_from_used_evidence(question.question, evidence_answer, used_articles)
        used_articles = finalize_evidence_answer(evidence_answer, evidence_blocks)
        answer_ms += (time.perf_counter() - repair_start) * 1000.0
        repair_attempts.append(
            {"stage": "answer_component_repair", "components": repairable_missing, "llm_calls": 1}
        )
        verification = verify_used_evidence_answer(
            evidence_answer.answer,
            used_articles,
            context_articles,
            question=question.question,
            required_components=plan.requested_components,
            target_norm_roles=plan.target_norm_roles,
            covered_components=evidence_answer.covered_components,
            insufficient_components=evidence_answer.insufficient_components,
            claims=evidence_answer.claims,
            evidence_by_id={block.evidence_id: block.article for block in evidence_blocks},
        )
    if not verification.ok and _repair_citations_to_used_evidence(evidence_answer, used_articles, verification.issues):
        verification = verify_used_evidence_answer(
            evidence_answer.answer,
            used_articles,
            context_articles,
            question=question.question,
            required_components=plan.requested_components,
            target_norm_roles=plan.target_norm_roles,
            covered_components=evidence_answer.covered_components,
            insufficient_components=evidence_answer.insufficient_components,
            claims=evidence_answer.claims,
            evidence_by_id={block.evidence_id: block.article for block in evidence_blocks},
        )
    pred = format_prediction(question, evidence_answer.answer, used_articles)
    trace = QuestionRunTrace(
        id=question.id,
        question=question.question,
        predicted_metadata=plan.predicted_metadata(),
        corpus_candidates=[candidate.to_dict() for candidate in corpus_candidates],
        candidate_counts=_retrieval_trace(searcher).get("candidate_counts", {}),
        reranked_evidence=[article.relevant_article for article in articles],
        supporting_spans=[str(article.metadata.get("support_span_label") or article.metadata.get("support_span_key") or "") for article in used_articles],
        retrieval_path=[
            key
            for key, value in _retrieval_trace(searcher).get("candidate_counts", {}).items()
            if value
        ],
        graph_expansions=[str(item) for item in _retrieval_trace(searcher).get("graph_expansions", [])],
        threshold_cutoff_reason=str(_retrieval_trace(searcher).get("threshold_cutoff_reason", "")),
        used_evidence_ids=evidence_answer.used_evidence_ids,
        verifier_issues=verification.issues,
        final_relevant_docs=pred.relevant_docs,
        final_relevant_articles=pred.relevant_articles,
        planner_ms=round(planner_ms, 2),
        retrieval_ms=round(retrieval_ms, 2),
        rerank_ms=float(_retrieval_trace(searcher).get("rerank_ms", 0.0) or 0.0),
        answer_ms=round(answer_ms, 2),
        planner_reasoning=bool(effective_planner_config.think),
        answer_reasoning=bool(effective_answer_config.think),
        planner_num_ctx=effective_planner_config.num_ctx,
        answer_num_ctx=effective_answer_config.num_ctx,
        actual_backend=_backend_name(searcher),
        repair_attempts=repair_attempts,
        evidence_metadata=build_evidence_metadata(used_articles),
        verification_warnings=list(verification.warnings or []),
    )
    return pred, trace, verification


def _component_coverage_missing(issues: list[str]) -> list[str]:
    prefix = "component_coverage_missing:"
    return [issue[len(prefix) :] for issue in issues if issue.startswith(prefix)]


def _context_supports_component(component: str, articles: list[ArticleNode]) -> bool:
    combined = " ".join(
        f"{article.title_for_submission} {article.article_title} {article.metadata.get('support_snippet', '')} {article.text[:1200]}".lower()
        for article in articles
    )
    return component_matches(component, combined, evidence=True)


def _adaptive_context_limit(plan: LegalQueryPlan, configured_limit: int | None) -> int | None:
    base = configured_limit or 5
    required_count = len(plan.requested_components)
    if plan.question_type == "comparison" or required_count >= 3:
        return min(max(base, 8), 8)
    if required_count >= 2 or (plan.needs_guidance_docs and plan.multi_hop_targets):
        return min(max(base, 7), 7)
    return min(base, 5)


def _adaptive_retrieval_top_k(plan: LegalQueryPlan, requested_top_k: int) -> int:
    if plan.question_type == "comparison" or len(plan.requested_components) >= 3:
        return max(requested_top_k, 8)
    if len(plan.requested_components) >= 2 or (plan.needs_guidance_docs and plan.multi_hop_targets):
        return max(requested_top_k, 7)
    return requested_top_k


def _adaptive_evidence_chars(plan: LegalQueryPlan) -> int:
    if plan.question_type == "comparison" or len(plan.requested_components) >= 2:
        return 1400
    if plan.needs_guidance_docs and plan.multi_hop_targets:
        return 1200
    return 900


def _select_evidence_candidates(
    plan: LegalQueryPlan,
    articles: list[ArticleNode],
    limit: int | None,
) -> list[ArticleNode]:
    if not articles:
        return []
    selected_limit = max(1, limit or len(articles))
    scored: list[tuple[float, int, ArticleNode]] = []
    required = list(dict.fromkeys(plan.requested_components))
    key_phrases = [
        *plan.must_keep_phrases[:4],
        *plan.entities.get("subjects", [])[:2],
        *plan.entities.get("actions", [])[:2],
        *plan.entities.get("objects", [])[:2],
    ]
    for rank, article in enumerate(articles):
        clone = ArticleNode.from_dict(article.to_dict())
        support = str(clone.metadata.get("support_snippet") or clone.text[:1800]).lower()
        component_hits = [component for component in required if component_matches(component, support, evidence=True)]
        phrase_hits = [phrase for phrase in key_phrases if phrase and phrase.lower() in support]
        support_type = _evidence_support_type(plan, clone, support)
        admission = float(clone.score) + 0.12 * len(component_hits) + 0.04 * len(phrase_hits)
        if support_type == "cross_reference":
            admission -= 0.12
        elif support_type == "adjacent":
            admission -= 0.18
        metadata = dict(clone.metadata)
        metadata["evidence_admission"] = {
            "score": round(admission, 6),
            "components": component_hits,
            "phrase_hits": phrase_hits,
            "support_type": support_type,
        }
        clone.metadata = metadata
        scored.append((admission, rank, clone))
    scored.sort(key=lambda item: (-item[0], item[1]))

    # Adjacent and cross-reference hits are useful during retrieval expansion,
    # but they must not become claimable answer evidence.  Keeping them out of
    # the generator context avoids asking the LLM to obey a distinction that
    # the deterministic pipeline can enforce itself.
    direct_scored = [
        item
        for item in scored
        if item[2].metadata.get("evidence_admission", {}).get("support_type") == "direct"
    ]

    output: list[ArticleNode] = []
    seen: set[str] = set()
    for component in required:
        for _score, _rank, article in direct_scored:
            admission = article.metadata.get("evidence_admission", {})
            if component in admission.get("components", []) and article.article_key not in seen:
                output.append(article)
                seen.add(article.article_key)
                break
    for _score, _rank, article in direct_scored:
        if article.article_key in seen:
            continue
        output.append(article)
        seen.add(article.article_key)
        if len(output) >= selected_limit:
            break
    return output[:selected_limit]


def _insufficient_answer_for_corpus_gap(plan: LegalQueryPlan, evidence_blocks: list[object]) -> EvidenceAnswer | None:
    if not plan.requested_components:
        return None
    if not evidence_blocks:
        components = list(dict.fromkeys(plan.requested_components))
        return EvidenceAnswer(
            answer=(
                "Chưa tìm thấy căn cứ trực tiếp trong corpus để trả lời đầy đủ về "
                f"{', '.join(components)}; không suy đoán từ văn bản dẫn chiếu hoặc quy định lân cận."
            ),
            used_evidence_ids=[],
            insufficient_evidence=True,
            insufficient_components=components,
            claims=[],
        )
    direct_blocks = [block for block in evidence_blocks if getattr(block, "support_type", "direct") == "direct"]
    direct_supports_requirement = any(
        component_matches(component, str(getattr(block, "support_snippet", "")), evidence=True)
        for block in direct_blocks
        for component in plan.requested_components
    )
    if direct_supports_requirement:
        return None
    components = list(dict.fromkeys(plan.requested_components))
    joined = ", ".join(components)
    return EvidenceAnswer(
        answer=f"Chưa tìm thấy căn cứ trực tiếp trong corpus để trả lời đầy đủ về {joined}; không suy đoán từ văn bản dẫn chiếu hoặc quy định lân cận.",
        used_evidence_ids=[],
        insufficient_evidence=True,
        insufficient_components=components,
        claims=[],
    )


def _looks_like_cross_reference_only(text: str) -> bool:
    pointer_markers = (
        "được thực hiện theo quy định",
        "thực hiện theo quy định",
        "theo quy định của pháp luật",
        "theo quy định tại điều",
        "đáp ứng quy định tại điều",
    )
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
    return any(marker in text for marker in pointer_markers) and not any(marker in text for marker in substantive_markers)


def _evidence_support_type(plan: LegalQueryPlan, article: ArticleNode, support: str) -> str:
    if _looks_like_cross_reference_only(support):
        return "cross_reference"
    question = plan.normalized_question.lower()
    title = article.article_title.lower()
    focus_phrases = _question_focus_phrases(plan)
    focus_scope = normalize_for_match(
        f"{article.title_for_submission} {article.article_title} {support} {article.text[:2400]}"
    )
    if focus_phrases and not any(_phrase_is_covered(phrase, focus_scope) for phrase in focus_phrases):
        return "adjacent"
    action_phrases = _question_action_phrases(plan)
    if action_phrases and not any(_phrase_is_covered(phrase, focus_scope) for phrase in action_phrases):
        return "adjacent"
    filing_question = "hồ sơ" in plan.requested_components or any(
        marker in question for marker in ("nộp đơn", "chuẩn bị", "tài liệu", "giấy tờ")
    )
    if any(marker in question for marker in ("nộp đơn đăng ký", "đơn đăng ký", "hồ sơ đăng ký")):
        filing_scope = f"{title} {support[:700]}"
        if not any(marker in filing_scope for marker in ("đơn đăng ký", "hồ sơ đăng ký", "yêu cầu đối với đơn")):
            return "adjacent"
    if filing_question and any(marker in title for marker in ("thẩm định", "công bố", "xử lý đơn", "kết quả")):
        return "adjacent"
    if "điều kiện" in plan.requested_components:
        condition_markers = ("điều kiện", "yêu cầu", "tiêu chí", "trường hợp")
        adjacent_markers = ("xâm phạm", "phạm vi quyền", "trách nhiệm", "xử lý vi phạm", "thủ tục")
        combined = f"{title} {support[:500]}"
        if any(marker in title for marker in adjacent_markers) and not any(
            marker in combined for marker in condition_markers
        ):
            return "adjacent"
        if "bảo hộ" in question:
            protection_markers = ("điều kiện bảo hộ", "được bảo hộ nếu", "được bảo hộ khi", "yêu cầu bảo hộ")
            if not any(marker in combined for marker in protection_markers):
                return "adjacent"
    return "direct"


def _question_focus_phrases(plan: LegalQueryPlan) -> list[str]:
    """Return specific, question-grounded legal concepts for evidence admission."""
    question = normalize_legal_query_text(normalize_for_match(plan.normalized_question))
    generic = {
        "công ty",
        "doanh nghiệp",
        "tài liệu",
        "thông tin",
        "điều kiện",
        "thủ tục",
        "hồ sơ",
        "trách nhiệm",
        "thẩm quyền",
        "quy định",
    }
    candidates = [
        *plan.entities.get("objects", []),
        *plan.entities.get("actions", []),
        *plan.legal_facets,
    ]
    output: list[str] = []
    for candidate in candidates:
        phrase = normalize_legal_query_text(normalize_for_match(candidate))
        words = tokenize(phrase)
        if not phrase or phrase in generic or not 2 <= len(words) <= 8:
            continue
        if phrase not in question:
            continue
        if phrase not in output:
            output.append(phrase)
    output.sort(key=lambda item: (-len(tokenize(item)), -len(item)))
    return output[:4]


def _question_action_phrases(plan: LegalQueryPlan) -> list[str]:
    question = normalize_legal_query_text(normalize_for_match(plan.normalized_question))
    generic_actions = {"áp dụng", "có", "được", "làm", "thực hiện", "xử lý"}
    output: list[str] = []
    for candidate in plan.entities.get("actions", []):
        phrase = normalize_legal_query_text(normalize_for_match(candidate))
        if not phrase or phrase in generic_actions or phrase not in question:
            continue
        if phrase not in output:
            output.append(phrase)
    return output[:3]


def _phrase_is_covered(phrase: str, scope: str) -> bool:
    phrase = normalize_legal_query_text(phrase)
    scope = normalize_legal_query_text(scope)
    if phrase in scope:
        return True
    phrase_tokens = [token for token in tokenize(phrase) if len(token) > 1]
    if len(phrase_tokens) < 2:
        return False
    scope_tokens = set(tokenize(scope))
    covered = sum(1 for token in phrase_tokens if token in scope_tokens)
    return covered / len(phrase_tokens) >= 0.8


def _append_insufficient_component_note(answer: str, components: list[str]) -> str:
    notes = []
    for component in components:
        notes.append(f"Về {component}, chưa tìm thấy căn cứ trực tiếp trong evidence được truy hồi.")
    suffix = " ".join(notes)
    if answer.endswith((".", "!", "?")):
        return f"{answer} {suffix}"
    return f"{answer}. {suffix}"


def _repair_list_answer_from_used_evidence(question: str, evidence_answer: object, used_articles: list[ArticleNode]) -> bool:
    lowered_question = question.lower()
    if "hồ sơ" not in lowered_question or not any(marker in lowered_question for marker in ("gồm", "bao gồm", "những gì")):
        return False
    if not used_articles:
        return False
    for article in used_articles:
        items = _best_lettered_items_for_question(question, article)
        if len(items) < 2:
            continue
        answer_lower = str(evidence_answer.answer).lower()
        covered = sum(1 for item in items if _item_is_covered(item, answer_lower))
        if covered >= max(2, len(items) - 1):
            return False
        joined = "; ".join(f"{label}) {text}" for label, text in items[:6])
        evidence_answer.answer = f"Theo {article.article_label} {article.doc_id}, hồ sơ gồm: {joined}."
        if "hồ sơ" not in evidence_answer.covered_components:
            evidence_answer.covered_components.append("hồ sơ")
        return True
    return False


def _repair_missing_components_from_direct_evidence(
    evidence_answer: object,
    evidence_blocks: list[object],
    missing_components: list[str],
) -> list[str]:
    repaired: list[str] = []
    for component in list(dict.fromkeys(missing_components)):
        if component_matches(component, str(evidence_answer.answer), evidence=True):
            continue
        support = _best_direct_component_sentence(component, evidence_blocks)
        if support is None:
            continue
        sentence, block = support
        if sentence.lower() not in str(evidence_answer.answer).lower():
            evidence_answer.answer = f"{str(evidence_answer.answer).rstrip()} {sentence}".strip()
        if component not in evidence_answer.covered_components:
            evidence_answer.covered_components.append(component)
        evidence_id = str(getattr(block, "evidence_id", ""))
        if evidence_id and evidence_id not in evidence_answer.used_evidence_ids:
            evidence_answer.used_evidence_ids.append(evidence_id)
        article = getattr(block, "article", None)
        claim = {
            "claim": sentence,
            "evidence_ids": [evidence_id] if evidence_id else [],
            "article_refs": [article.relevant_article] if article is not None else [],
        }
        evidence_answer.claims.append(claim)
        evidence_answer.support_map = evidence_answer.claims
        repaired.append(component)
    return repaired


def _best_direct_component_sentence(component: str, evidence_blocks: list[object]) -> tuple[str, object] | None:
    candidates: list[tuple[int, int, str, object]] = []
    for block_index, block in enumerate(evidence_blocks):
        if str(getattr(block, "support_type", "direct")) != "direct":
            continue
        article = getattr(block, "article", None)
        sources = [str(getattr(block, "support_snippet", "") or "")]
        if article is not None:
            sources.append(str(getattr(article, "text", "") or ""))
        for source in sources:
            for sentence in re.split(r"(?<=[.;!?])\s+|\n+", source):
                cleaned = " ".join(sentence.split()).strip(" ;")
                if not 20 <= len(cleaned) <= 700 or not component_matches(component, cleaned, evidence=True):
                    continue
                candidates.append((len(cleaned), block_index, cleaned, block))
    if not candidates:
        return None
    _length, _index, sentence, block = min(candidates, key=lambda item: (item[0], item[1]))
    if sentence[-1:] not in ".!?":
        sentence += "."
    return sentence, block


def _best_lettered_items_for_question(question: str, article: ArticleNode) -> list[tuple[str, str]]:
    candidates: list[tuple[float, list[tuple[str, str]]]] = []
    for source in _list_sources(article):
        items = _extract_lettered_items(source)
        if len(items) < 2:
            continue
        candidates.append((_list_source_score(question, source), items))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _list_sources(article: ArticleNode) -> list[str]:
    sources: list[str] = []
    snippet = str(article.metadata.get("support_snippet") or "").strip()
    if snippet:
        sources.append(snippet)
    text = str(article.text or "").strip()
    if text:
        clause_sources = _numbered_clause_sources(text)
        for source in clause_sources or [text]:
            if source and source not in sources:
                sources.append(source)
    for clause in article.metadata.get("clause_nodes", []) or []:
        text = str(clause.get("text") or "").strip()
        if text and text not in sources:
            sources.append(text)
    return sources


def _numbered_clause_sources(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(r"(?:^|\n)(\d+)\.\s+(.+?)(?=(?:\n\d+\.\s+)|\Z)", re.DOTALL)
    output: list[str] = []
    for number, body in pattern.findall(normalized):
        source = f"{number}. {body}".strip()
        if len(source) >= 20:
            output.append(source)
    return output


def _list_source_score(question: str, source: str) -> float:
    lowered_question = question.lower()
    lowered = source.lower()
    score = 0.0
    if "hồ sơ" in lowered:
        score += 2.0
    if re.search(r"^\s*\d+\.\s+hồ sơ\b.+\bbao gồm\b", lowered):
        score += 6.0
    elif "bao gồm" in lowered and "hồ sơ" in lowered:
        score += 1.0
    if any(term in lowered_question for term in ("đề nghị", "đề xuất", "nhu cầu hỗ trợ")):
        if any(term in lowered for term in ("hồ sơ đề xuất nhu cầu hỗ trợ", "đề xuất nhu cầu hỗ trợ")):
            score += 4.0
        if "hồ sơ thanh toán" in lowered:
            score -= 3.0
        if "trong thời hạn" in lowered and "xem xét hồ sơ" in lowered and not re.search(r"^\s*\d+\.\s+hồ sơ\b.+\bbao gồm\b", lowered):
            score -= 5.0
    if "cụm liên kết ngành" in lowered_question and "cụm liên kết ngành" in lowered:
        score += 1.0
    score += min(len(_extract_lettered_items(source)), 6) * 0.25
    return score


def _extract_lettered_items(text: str) -> list[tuple[str, str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(
        r"(?:^|\n)\s*([a-zđ])\)\s+(.+?)(?=(?:\n\s*(?:[a-zđ]\)\s+|\d+\.\s+))|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    items: list[tuple[str, str]] = []
    for label, body in pattern.findall(normalized):
        cleaned = " ".join(body.split()).strip(" ;.")
        if not cleaned:
            continue
        items.append((label.lower(), cleaned[:260]))
    return items


def _item_is_covered(item: tuple[str, str], answer_lower: str) -> bool:
    _label, text = item
    generic = {
        "theo",
        "định",
        "nghị",
        "doanh",
        "nghiệp",
        "nhỏ",
        "vừa",
        "hỗ",
        "trợ",
        "hồ",
        "sơ",
        "đề",
        "xuất",
        "nhu",
        "cầu",
        "liên",
        "quan",
        "nội",
        "dung",
    }
    tokens = [
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text.lower())
        if len(token) >= 4 and token not in generic
    ]
    return any(token in answer_lower for token in tokens[:8])


def _repair_citations_to_used_evidence(evidence_answer: object, used_articles: list[ArticleNode], issues: list[str]) -> bool:
    if len(used_articles) != 1:
        return False
    bad_labels = [
        issue.split(":", 1)[1]
        for issue in issues
        if issue.startswith("citation_not_in_used_evidence:") and ":" in issue
    ]
    article = used_articles[0]
    canonical = article.article_label
    answer = str(evidence_answer.answer)
    changed = False
    if bad_labels:
        evidence_text = f"{article.metadata.get('support_snippet', '')} {article.text}".lower()
        if any(bad_label.lower() not in evidence_text for bad_label in bad_labels):
            return False
        for bad_label in bad_labels:
            pattern = re.compile(re.escape(bad_label), re.IGNORECASE)
            answer, count = pattern.subn(canonical, answer)
            changed = changed or count > 0
    elif "answer_missing_article_citation" not in issues:
        return False
    if canonical.lower() not in answer.lower():
        suffix = f"Căn cứ {canonical} {article.doc_id}."
        answer = f"{answer} {suffix}" if answer.endswith((".", "!", "?")) else f"{answer}. {suffix}"
        changed = True
    if changed:
        evidence_answer.answer = answer
    return changed


def _cmd_eval_retrieval(
    questions_path: str,
    expected_path: str,
    index_path: str,
    articles_path: str,
    backend: str,
    config: dict,
    top_k: int,
) -> int:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    expected_raw = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    expected = {
        int(row["id"]): {str(value) for value in row.get("relevant_articles", [])}
        for row in expected_raw
    }
    try:
        searcher, manifest = _load_retrieval_backend(backend, index_path, articles_path, config)
    except HybridRetrievalError as exc:
        print(f"retrieval backend failed: {exc}")
        return 1
    report = evaluate_retrieval(questions, expected, lambda q, k: _search_articles(searcher, q, k), top_k)
    print(json.dumps({"retrieval": manifest, **report}, ensure_ascii=False, indent=2))
    return 0


def _cmd_eval_pipeline(
    questions_path: str,
    expected_path: str,
    index_path: str,
    articles_path: str,
    config: dict,
    top_k: int,
    planner_config: OllamaConfig,
) -> int:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    gold_metadata_by_id, gold_docs_by_id, gold_articles_by_id = load_gold_annotations(expected_path)
    bm25 = BM25Index.load(index_path)
    try:
        hybrid, hybrid_manifest = _load_retrieval_backend("hybrid_qdrant", index_path, articles_path, config)
    except HybridRetrievalError as exc:
        print(f"retrieval backend failed: {exc}")
        return 1

    planner_cache: dict[str, LegalQueryPlan] = {}
    lexicon = _load_lexicon_from_config(config)
    question_objects = [Question(id=int(item["id"]), question=str(item["question"])) for item in questions]
    predicted_metadata_by_id: dict[int, PredictedQuestionMetadata] = {}
    planner_predictions: list = []
    planner_traces: list[QuestionRunTrace] = []

    def planned_search(question: str, k: int):
        plan = planner_cache.get(question)
        if plan is None:
            plan = plan_legal_query(question, planner_config, lexicon_candidates=_lexicon_matches(lexicon, question))
            planner_cache[question] = plan
        return _search_articles(hybrid, question, k, plan=plan)

    def planned_search_multihop(question: str, k: int):
        plan = planner_cache.get(question)
        if plan is None:
            plan = plan_legal_query(question, planner_config, lexicon_candidates=_lexicon_matches(lexicon, question))
            planner_cache[question] = plan
        if plan.needs_guidance_docs and not plan.multi_hop_targets:
            plan.multi_hop_targets = ["Nghị định", "Thông tư"]
        return _search_articles(hybrid, question, k, plan=plan)

    try:
        for question in question_objects:
            pred, trace, _verification = _answer_question(
                question,
                hybrid,
                top_k,
                planner_config,
                planner_config,
                max_context_articles=_max_context_articles(config),
                lexicon=lexicon,
            )
            planner_predictions.append(pred)
            planner_traces.append(trace)
            predicted_metadata_by_id[question.id] = trace.predicted_metadata
        report = {
            "bm25_exact": evaluate_retrieval(questions, gold_articles_by_id, lambda q, k: retrieve_articles(bm25, q, top_k=k), top_k),
            "hybrid_qdrant": evaluate_retrieval(questions, gold_articles_by_id, lambda q, k: _search_articles(hybrid, q, k), top_k),
            "planner_hybrid": evaluate_retrieval(questions, gold_articles_by_id, planned_search, top_k),
            "planner_hybrid_multihop": evaluate_retrieval(questions, gold_articles_by_id, planned_search_multihop, top_k),
        }
    except (HybridRetrievalError, OllamaError) as exc:
        print(f"eval pipeline failed: {exc}")
        return 1
    print(
        json.dumps(
            {
                "planner_backend": "ollama",
                "planner_model": planner_config.model,
                "retrieval": hybrid_manifest,
                "metadata_metrics": compute_metadata_metrics(predicted_metadata_by_id, gold_metadata_by_id),
                "task_metrics": compute_prediction_metrics(
                    build_prediction_map(planner_predictions),
                    gold_docs_by_id,
                    gold_articles_by_id,
                    build_trace_map(planner_traces),
                ),
                **report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_validate(input_path: str, questions_path: str | None, max_issues: int) -> int:
    issues = validate_submission(input_path, questions_path)
    for issue in issues[:max_issues]:
        print(issue)
    if len(issues) > max_issues:
        print(f"... truncated {len(issues) - max_issues} more issues")
    print(f"validation issues: {len(issues)}")
    return 1 if issues else 0


def _cmd_package(input_path: str, output_path: str, allow_verifier_issues: bool = False) -> int:
    issues = validate_submission(input_path)
    issues.extend(validate_package_manifest(input_path, allow_verifier_issues=allow_verifier_issues))
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


def _load_retrieval_backend(
    backend: str,
    index_path: str,
    articles_path: str,
    config: dict,
) -> tuple[object, dict]:
    normalized = _normalize_backend_name(backend)
    if normalized == "bm25_exact":
        return BM25Index.load(index_path), {"retrieval_backend": "bm25_exact", "actual_backend": "bm25_exact"}
    if normalized == "fts5_bm25":
        lexical_path = str(config.get("retrieval", {}).get("lexical_index") or index_path)
        return FTS5Index.load(lexical_path), {"retrieval_backend": "fts5_bm25", "actual_backend": "fts5_bm25"}
    if normalized == "hybrid_qdrant":
        hybrid_config = config_from_mapping(config)
        _ensure_hybrid_runtime_ready(hybrid_config, config)
        retriever = HybridRetriever.load(articles_path, index_path, hybrid_config)
        return (
            retriever,
            {
                "retrieval_backend": "hybrid_qdrant",
                "actual_backend": "hybrid_qdrant",
                "qdrant_url": hybrid_config.qdrant_url,
                "qdrant_path": hybrid_config.qdrant_path,
                "qdrant_collection": hybrid_config.collection,
                "embedding_model": hybrid_config.embedding_model,
                "reranker_model": hybrid_config.reranker_model,
                "bm25_top_k": hybrid_config.bm25_top_k,
                "vector_top_k": hybrid_config.vector_top_k,
                "fusion_top_k": hybrid_config.fusion_top_k,
                "rerank_top_k": hybrid_config.rerank_top_k,
                "local_files_only": hybrid_config.local_files_only,
            },
        )
    raise HybridRetrievalError(f"unknown_retrieval_backend:{backend}")


def _ensure_hybrid_runtime_ready(hybrid_config: HybridRetrievalConfig, config: dict) -> None:
    report = _load_hybrid_report(config)
    points = int(report.get("points") or 0) if report else 0
    if hybrid_config.qdrant_path:
        if points > hybrid_config.max_embedded_points:
            raise HybridRetrievalError(
                "embedded_qdrant_too_large_for_runtime:"
                f"points={points}:limit={hybrid_config.max_embedded_points}:"
                "start_qdrant_server_and_set_qdrant.path_empty"
            )
        return
    report_collection = str(report.get("collection") or "") if report else ""
    if report_collection and report_collection != hybrid_config.collection:
        raise HybridRetrievalError(
            "qdrant_collection_report_mismatch:"
            f"configured={hybrid_config.collection}:reported={report_collection}"
        )
    if not _qdrant_server_collection_ready(hybrid_config.qdrant_url, hybrid_config.collection):
        raise HybridRetrievalError(
            "qdrant_server_collection_unavailable:"
            f"{hybrid_config.qdrant_url.rstrip('/')}/collections/{hybrid_config.collection}"
        )


def _load_hybrid_report(config: dict) -> dict:
    report_path = Path(config.get("retrieval", {}).get("hybrid_report", "data/indices/hybrid_index_report.json"))
    if not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _search_articles(searcher: object, question: str, top_k: int, plan: LegalQueryPlan | None = None) -> list:
    if isinstance(searcher, FTS5Index):
        exact_text = question if plan is None else " ".join(
            [question, plan.normalized_question, *plan.target_doc_ids, *plan.target_article_labels]
        )
        exact_hits = searcher.exact_search(exact_text, top_k=max(top_k, 10))
        scored = {article.article_key: article for article in exact_hits}
        queries = [question] if plan is None else [query.text for query in plan.queries if query.kind != "exact"] or [question]
        for query in queries:
            for article in searcher.search(query, top_k=max(top_k * 4, 20)):
                current = scored.get(article.article_key)
                if current is None or article.score > current.score:
                    scored[article.article_key] = article
        hits = sorted(scored.values(), key=lambda article: article.score, reverse=True)[:top_k]
        searcher.last_trace = {
            "candidate_counts": {"exact": len(exact_hits), "fts": len(scored)},
            "fused": len(scored),
            "reranked": len(hits),
            "final": len(hits),
            "actual_backend": "fts5_bm25",
        }
        return hits
    if isinstance(searcher, BM25Index):
        if plan is None:
            hits = retrieve_articles(searcher, question, top_k=top_k)
            searcher.last_trace = {
                "candidate_counts": {"bm25": len(hits)},
                "fused": len(hits),
                "reranked": len(hits),
                "final": len(hits),
                "actual_backend": "bm25_exact",
            }
            return hits
        scored = {}
        queries = [query.text for query in plan.queries] or [question]
        exact_text = " ".join([question, plan.normalized_question, *plan.target_doc_ids, *plan.target_article_labels])
        for article in retrieve_articles(searcher, exact_text, top_k=max(top_k, 20), min_score_ratio=0.0):
            article.metadata = {**article.metadata, "retrieval_trace": f"query_kind=planner_targets; source=bm25_exact; score={article.score:.4f}"}
            scored[article.article_key] = article
        candidate_counts = {"exact": len(scored)}
        for query in queries:
            before = len(scored)
            for article in retrieve_articles(searcher, query, top_k=max(top_k, 20), min_score_ratio=0.0):
                article.metadata = {**article.metadata, "retrieval_trace": f"query={query}; source=bm25; score={article.score:.4f}"}
                current = scored.get(article.article_key)
                if current is None or article.score > current.score:
                    scored[article.article_key] = article
            candidate_counts[f"bm25:{query}"] = len(scored) - before if len(scored) > before else 0
        biased_hits = sorted(scored.values(), key=lambda article: article.score, reverse=True)
        for article in biased_hits:
            title = f"{article.title_for_submission} {article.article_title} {article.text[:400]}".lower()
            must_terms = [term.lower() for term in plan.filters.get("must_include_terms", []) if term]
            should_terms = [term.lower() for term in plan.filters.get("should_include_terms", []) if term]
            governing_hints = [hint.lower() for hint in plan.governing_doc_hints if hint]
            target_roles = {role.lower() for role in plan.target_norm_roles}
            norm_roles = {str(role).lower() for role in article.metadata.get("norm_roles", [])}
            if must_terms and not any(term in title for term in must_terms):
                article.score -= 1.0
            article.score += 0.08 * sum(1 for term in should_terms if term in title)
            article.score += 0.18 * sum(1 for hint in governing_hints if hint in title)
            if target_roles and norm_roles & target_roles:
                article.score += 0.24 * len(norm_roles & target_roles)
        hits = sorted(biased_hits, key=lambda article: article.score, reverse=True)[:top_k]
        searcher.last_trace = {
            "candidate_counts": candidate_counts,
            "fused": len(scored),
            "reranked": len(hits),
            "final": len(hits),
            "actual_backend": "bm25_exact",
        }
        return hits
    if isinstance(searcher, HybridRetriever):
        if plan is not None:
            return searcher.search_with_plan(question, plan, top_k=top_k)
        return searcher.search(question, top_k=top_k)
    raise HybridRetrievalError(f"unsupported_searcher:{type(searcher).__name__}")


def _retrieval_trace(searcher: object) -> dict:
    return dict(getattr(searcher, "last_trace", {}) or {})


def _resolve_corpus_candidates(searcher: object, question: str):
    resolver = getattr(searcher, "resolve_corpus_candidates", None)
    if not callable(resolver):
        return []
    try:
        return resolver(question, max_docs=8)
    except (OSError, RuntimeError, ValueError):
        return []


def _load_lexicon_from_config(config: dict) -> list[LegalLexiconEntry] | LegalLexiconIndex:
    lexicon_cfg = config.get("lexicon", {})
    path = lexicon_cfg.get("path")
    if not path:
        return []
    return load_legal_lexicon(path)


def _lexicon_matches(lexicon: list[LegalLexiconEntry] | LegalLexiconIndex | None, question: str):
    if not lexicon:
        return []
    return search_legal_lexicon(lexicon, question, limit=4)


def _retry_retrieval_if_drift(
    searcher: object,
    question: str,
    top_k: int,
    plan: LegalQueryPlan,
    articles: list[ArticleNode],
) -> tuple[list[ArticleNode] | None, str]:
    issue = _retrieval_critic_issue(plan, articles)
    if not issue:
        return None, ""
    retry_query = _critic_retry_query(question, plan)
    if not retry_query:
        return None, ""
    retry_plan = LegalQueryPlan(
        intent=plan.intent,
        question_scope=plan.question_scope,
        normalized_question=plan.normalized_question,
        question_type=plan.question_type,
        answer_shape=plan.answer_shape,
        legal_terms=plan.legal_terms,
        legal_facets=plan.legal_facets,
        requested_components=plan.requested_components,
        target_norm_roles=plan.target_norm_roles,
        governing_doc_hints=plan.governing_doc_hints,
        domain_anchors=plan.domain_anchors,
        lexical_expansions=plan.lexical_expansions,
        candidate_regimes=plan.candidate_regimes,
        must_keep_phrases=plan.must_keep_phrases,
        entities=plan.entities,
        target_doc_ids=plan.target_doc_ids,
        target_doc_aliases=plan.target_doc_aliases,
        target_article_labels=plan.target_article_labels,
        queries=[*plan.queries, PlannedQuery("component", retry_query, "retrieval critic retry")],
        filters=plan.filters,
        retrieval_bias=plan.retrieval_bias,
        needs_guidance_docs=plan.needs_guidance_docs,
        multi_hop_targets=plan.multi_hop_targets,
        missing_facts=plan.missing_facts,
        confidence=plan.confidence,
    )
    retry_articles = _search_articles(searcher, question, top_k, plan=retry_plan)
    if _coverage_score(plan, retry_articles) > _coverage_score(plan, articles):
        return retry_articles, issue
    return None, ""


def _targeted_component_retrieval(
    searcher: object,
    question: str,
    top_k: int,
    plan: LegalQueryPlan,
    current_articles: list[ArticleNode],
    component: str,
) -> list[ArticleNode]:
    if isinstance(searcher, HybridRetriever):
        return searcher.search_targeted_component(question, plan, component, current_articles, top_k=top_k)

    query_parts = [
        *plan.must_keep_phrases[:3],
        *plan.entities.get("subjects", [])[:2],
        *plan.entities.get("actions", [])[:2],
        component,
        *plan.target_doc_ids[:1],
        *plan.target_article_labels[:1],
        *plan.lexical_expansions[:2],
    ]
    targeted_query = " ".join(" ".join(str(item) for item in query_parts if str(item).strip()).split()[:30])
    targeted_plan = replace(
        plan,
        queries=[PlannedQuery("component", targeted_query or f"{question} {component}", f"repair required component: {component}", [component])],
        requested_components=[component],
    )
    targeted = _search_articles(searcher, question, max(top_k * 2, 10), plan=targeted_plan)
    merged: dict[str, ArticleNode] = {}
    for rank, article in enumerate([*targeted, *current_articles], start=1):
        clone = ArticleNode.from_dict(article.to_dict())
        clone.score = max(float(clone.score), 1.0 / rank)
        existing = merged.get(clone.article_key)
        if existing is None or clone.score > existing.score:
            merged[clone.article_key] = clone
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]


def _targeted_components_retrieval(
    searcher: object,
    question: str,
    top_k: int,
    plan: LegalQueryPlan,
    current_articles: list[ArticleNode],
    components: list[str],
) -> list[ArticleNode]:
    selected = list(dict.fromkeys(component for component in components if component))[:3]
    if not selected:
        return current_articles
    if isinstance(searcher, HybridRetriever):
        return searcher.search_targeted_components(
            question,
            plan,
            selected,
            current_articles,
            top_k=top_k,
        )
    articles = current_articles
    for component in selected:
        articles = _targeted_component_retrieval(searcher, question, top_k, plan, articles, component)
    return articles


def _retrieval_critic_issue(plan: LegalQueryPlan, articles: list[ArticleNode]) -> str:
    if not articles:
        return "no_articles"
    missing_components = [component for component in plan.requested_components if not _context_supports_component(component, articles)]
    if missing_components:
        return f"missing_components:{'|'.join(missing_components[:3])}"
    phrases = [phrase for phrase in [*plan.must_keep_phrases[:4], *plan.lexical_expansions[:4]] if len(phrase) > 3]
    if phrases and not _context_covers_any(articles, phrases):
        return "missing_lexical_anchor"
    regimes = [item for item in plan.candidate_regimes[:5] if "/" in item or len(item) > 8]
    if regimes and not _context_covers_any(articles, regimes):
        return "missing_candidate_regime"
    return ""


def _coverage_score(plan: LegalQueryPlan, articles: list[ArticleNode]) -> float:
    score = 0.0
    for component in plan.requested_components:
        if _context_supports_component(component, articles):
            score += 2.0
    combined = _combined_context(articles)
    for phrase in [*plan.must_keep_phrases, *plan.lexical_expansions[:8], *plan.candidate_regimes[:6]]:
        if phrase and phrase.lower() in combined:
            score += 1.0
    return score


def _context_covers_any(articles: list[ArticleNode], phrases: list[str]) -> bool:
    combined = _combined_context(articles)
    return any(phrase.lower() in combined for phrase in phrases)


def _combined_context(articles: list[ArticleNode]) -> str:
    return " ".join(
        f"{article.doc_id} {article.title_for_submission} {article.article_title} {article.metadata.get('support_snippet', '')} {article.text[:1600]}".lower()
        for article in articles
    )


def _critic_retry_query(question: str, plan: LegalQueryPlan) -> str:
    phrases = [
        *plan.must_keep_phrases[:4],
        *plan.lexical_expansions[:6],
        *plan.requested_components[:4],
        *plan.candidate_regimes[:4],
        *plan.governing_doc_hints[:3],
        *plan.legal_facets[:4],
    ]
    output: list[str] = []
    seen: set[str] = set()
    for phrase in phrases or [question]:
        text = " ".join(str(phrase).split()).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return " ".join(" ".join(output).split()[:28])


def _planner_config(config: dict, model: str | None, ollama_url: str, max_tokens: int) -> OllamaConfig:
    planning = config.get("planning", {})
    generation = config.get("generation", {})
    return OllamaConfig(
        model=model or planning.get("model") or generation.get("target_model") or DEFAULT_OLLAMA_MODEL,
        url=planning.get("ollama_url") or ollama_url or generation.get("ollama_url") or DEFAULT_OLLAMA_URL,
        max_tokens=int(planning.get("max_tokens", max_tokens)),
        response_format_json=bool(planning.get("response_format_json", True)),
        keep_alive=str(planning.get("keep_alive", "30m")),
        think=bool(planning.get("think", False)),
        num_ctx=int(planning.get("num_ctx", 8192)),
    )


def _answer_config(config: dict, model: str | None, ollama_url: str, max_tokens: int) -> OllamaConfig:
    generation = config.get("generation", {})
    return OllamaConfig(
        model=model or generation.get("target_model") or DEFAULT_OLLAMA_MODEL,
        url=generation.get("ollama_url") or ollama_url or DEFAULT_OLLAMA_URL,
        max_tokens=int(generation.get("max_tokens", max_tokens)),
        response_format_json=bool(generation.get("response_format_json", True)),
        keep_alive=str(generation.get("keep_alive", "30m")),
        think=generation.get("think", False),
        num_ctx=int(generation.get("num_ctx", 12288)),
    )


def _max_context_articles(config: dict) -> int | None:
    value = config.get("generation", {}).get("max_context_articles")
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_backend(cli_backend: str | None, config: dict) -> str:
    if cli_backend:
        return cli_backend
    return config.get("retrieval", {}).get("backend", "bm25_exact")


def _resolve_articles_path(cli_articles: str, config: dict) -> str:
    return cli_articles or config.get("retrieval", {}).get("articles", "data/normalized/articles.jsonl")


def _resolve_index_path(cli_index: str | None, config: dict) -> str:
    return cli_index or config.get("retrieval", {}).get("bm25_index", "data/indices/bm25_index.json")


def _normalize_backend_name(backend: str) -> str:
    if backend in {"bm25", "bm25_baseline", "bm25_exact"}:
        return "bm25_exact"
    if backend in {"fts", "fts5", "fts5_bm25"}:
        return "fts5_bm25"
    return backend


def _backend_name(searcher: object) -> str:
    if isinstance(searcher, HybridRetriever):
        return "hybrid_qdrant"
    if isinstance(searcher, BM25Index):
        return "bm25_exact"
    if isinstance(searcher, FTS5Index):
        return "fts5_bm25"
    return type(searcher).__name__


def _hybrid_backend_ready(config: dict) -> bool:
    report_path = Path(config.get("retrieval", {}).get("hybrid_report", "data/indices/hybrid_index_report.json"))
    if not report_path.exists():
        return False
    qdrant_path = str(config.get("qdrant", {}).get("path", "") or "")
    if qdrant_path:
        path = Path(qdrant_path)
        if not path.exists():
            return False
        files = [item for item in path.rglob("*") if item.is_file()]
        return len(files) > 2
    qdrant_url = str(config.get("qdrant", {}).get("url", "http://127.0.0.1:6333") or "")
    collection = str(config.get("qdrant", {}).get("collection", "r2ai_law_articles_v2") or "")
    return _qdrant_server_collection_ready(qdrant_url, collection)


def _qdrant_server_collection_ready(url: str, collection: str) -> bool:
    try:
        with urlopen(f"{url.rstrip('/')}/collections/{collection}", timeout=5) as response:
            if not 200 <= response.status < 300:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            result = payload.get("result") or {}
            return result.get("status") == "green" and int(result.get("points_count") or 0) > 0
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return False


def _cmd_submit(args: argparse.Namespace) -> int:
    config = _load_config(getattr(args, "config", None))
    questions_path = config.get("questions", args.questions)
    index_path = _resolve_index_path(config.get("index", args.index), config)
    articles_path = _resolve_articles_path(args.articles, config)
    backend = _resolve_backend(args.backend, config)
    model = config.get("generation", {}).get("target_model", args.model)
    ollama_url = config.get("generation", {}).get("ollama_url", args.ollama_url)
    max_tokens = args.max_tokens
    top_k = config.get("retrieval", {}).get("final_top_k", args.top_k)
    output_dir = Path(args.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_{timestamp}.json"
    canonical_path = output_dir / "results.json"
    zip_path = output_dir / f"submission_{timestamp}.zip"

    print(f"=== R2AI Legal RAG Submission ===")
    print(f"timestamp : {timestamp}")
    print(f"questions : {questions_path}")
    print(f"index     : {index_path}")
    print(f"backend   : {backend}")
    print(f"model     : {model}")
    print(f"output    : {results_path}")
    print()

    ollama_config = _answer_config(config, model, ollama_url, max_tokens)
    code = _cmd_run_batch(
        questions_path,
        index_path,
        str(results_path),
        top_k,
        ollama_config,
        backend=backend,
        articles_path=articles_path,
        config=config,
        allow_verifier_issues=args.allow_verifier_issues,
    )
    if code != 0:
        return code

    shutil.copy2(results_path, canonical_path)
    ts_manifest = results_path.with_suffix(".manifest.json")
    canonical_manifest = canonical_path.with_suffix(".manifest.json")
    if ts_manifest.exists():
        shutil.copy2(ts_manifest, canonical_manifest)
    print(f"copied to {canonical_path}")

    issues = validate_submission(str(canonical_path), questions_path)
    if issues:
        for issue in issues[:20]:
            print(f"  validation: {issue}")
        print(f"validation issues: {len(issues)} (submission may still be usable)")

    pkg_issues = validate_package_manifest(str(canonical_path), allow_verifier_issues=args.allow_verifier_issues)
    if pkg_issues:
        for issue in pkg_issues:
            print(f"  package: {issue}")
        print("skipping zip packaging due to manifest issues")
    else:
        package_submission(str(canonical_path), str(zip_path))
        print(f"packaged: {zip_path}")

    print(f"\n=== Done ===")
    print(f"results : {results_path}")
    print(f"canonical: {canonical_path}")
    if zip_path.exists():
        print(f"zip     : {zip_path}")
    return 0


def _load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        print(f"warning: config {path} not found, using defaults")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
