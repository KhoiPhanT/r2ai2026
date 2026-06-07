from __future__ import annotations

import argparse
import json
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path

from legal_rag.corpus.ingest import ingest_corpus, read_articles_jsonl, write_articles_jsonl
from legal_rag.corpus.vbpl import (
    VbplImportConfig,
    clean_generated_data,
    import_vbpl_corpus,
    inspect_vbpl_corpus,
)
from legal_rag.documents import normalize_documents
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
    articles_from_used_evidence,
    build_evidence_blocks,
    generate_evidence_answer,
    OllamaConfig,
    OllamaError,
    generate_grounded_answer,
)
from legal_rag.planner import LegalQueryPlan, fallback_plan_for_debug, plan_legal_query
from legal_rag.question_metadata import split_questions_for_gold
from legal_rag.retrieval import (
    BM25Index,
    HybridRetrievalError,
    HybridRetriever,
    build_hybrid_index,
    config_from_mapping,
    evaluate_retrieval,
    retrieve_articles,
)
from legal_rag.schemas.models import PredictedQuestionMetadata, Question, QuestionRunTrace
from legal_rag.verifier import verify_prediction_evidence
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

    hybrid_build = subparsers.add_parser("build_hybrid_index")
    hybrid_build.add_argument("--input", required=True)
    hybrid_build.add_argument("--config", default="configs/local_m4.json")
    hybrid_build.add_argument("--report", default="data/indices/hybrid_index_report.json")

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
    if args.command == "build_hybrid_index":
        return _cmd_build_hybrid_index(args.input, args.config, args.report)
    if args.command == "plan_query":
        config = _load_config(args.config)
        planner_config = _planner_config(config, args.model, args.ollama_url, args.max_tokens)
        return _cmd_plan_query(args.question, planner_config)
    if args.command == "prepare_gold_metadata":
        config = _load_config(args.config)
        planner_config = _planner_config(config, args.model, args.ollama_url, args.max_tokens)
        return _cmd_prepare_gold_metadata(args.questions, args.output_dir, planner_config)
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


def _cmd_plan_query(question: str, planner_config: OllamaConfig) -> int:
    try:
        plan = plan_legal_query(question, planner_config)
    except OllamaError as exc:
        print(f"planner failed: {exc}")
        return 1
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_prepare_gold_metadata(questions_path: str, output_dir: str, planner_config: OllamaConfig) -> int:
    questions = load_questions(questions_path)
    predicted_by_id: dict[int, PredictedQuestionMetadata] = {}
    planner_failures: list[int] = []
    for question in questions:
        try:
            plan = plan_legal_query(question.question, planner_config)
        except OllamaError:
            planner_failures.append(question.id)
            plan = fallback_plan_for_debug(question.question)
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
) -> int:
    questions = load_questions(questions_path)
    try:
        searcher, retrieval_manifest = _load_retrieval_backend(backend, index_path, articles_path, config)
    except HybridRetrievalError as exc:
        print(f"retrieval backend failed: {exc}")
        return 1
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out.with_suffix(".progress.jsonl")

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
    batch_start = time.time()
    planner_config = _planner_config(config, ollama_config.model, ollama_config.url, 900)

    # --- Stream: process remaining questions one by one ---
    with progress_path.open("a", encoding="utf-8") as progress_file:
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
                )
            except (OllamaError, HybridRetrievalError) as exc:
                elapsed = time.time() - batch_start
                print(f"\n[{done_count}/{total}] FAIL id={question.id}: {exc} ({elapsed:.0f}s elapsed)")
                print("run_batch is fail-closed; no final results.json will be written.")
                return 1

            if not verification.ok:
                verifier_issues.append({"id": question.id, "issues": verification.issues})
                if not allow_verifier_issues:
                    print(f"\n[{done_count}/{total}] verifier failed id={question.id}: {verification.issues}")
                    print("run_batch is fail-closed; no final results.json will be written.")
                    return 1

            pred_dict = pred.to_dict()
            trace_rows.append(trace)

            # Append to progress file immediately
            progress_file.write(json.dumps(pred_dict, ensure_ascii=False) + "\n")
            progress_file.flush()

            done_by_id[question.id] = pred_dict

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

    out.write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_report(
        out.with_suffix(".manifest.json"),
        {
            "questions": len(questions),
            "completed": len(all_predictions),
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
    print(f"\nwrote {len(all_predictions)} predictions to {output_path} ({elapsed:.0f}s)")
    if verifier_issues:
        print(f"warning: {len(verifier_issues)} verifier issues (see manifest)")
    if len(all_predictions) < total:
        print(f"warning: {total - len(all_predictions)} questions have no prediction")
    return 0


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
        )
    except (OllamaError, HybridRetrievalError) as exc:
        print(f"model generation failed: {exc}")
        return 1

    print(pred.answer)
    print("\nPredicted metadata:")
    print(json.dumps(trace.predicted_metadata.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nBackend thực tế: {trace.actual_backend or _backend_name(searcher)}")
    print(
        f"Stage timing: planner={trace.planner_ms:.0f}ms retrieval={trace.retrieval_ms:.0f}ms answer={trace.answer_ms:.0f}ms"
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
        plan = plan_legal_query(question, _planner_config(config, ollama_config.model, ollama_config.url, ollama_config.max_tokens))
        articles = _search_articles(searcher, question, top_k, plan=plan)
    except (HybridRetrievalError, OllamaError) as exc:
        print(f"debug pipeline failed: {exc}")
        return 1
    blocks = build_evidence_blocks(articles)
    print(
        json.dumps(
            {
                "retrieval": retrieval_manifest,
                "plan": plan.to_dict(),
                "predicted_metadata": plan.predicted_metadata().to_dict(),
                "actual_backend": _retrieval_trace(searcher).get("actual_backend", _backend_name(searcher)),
                "candidate_counts": _retrieval_trace(searcher).get("candidate_counts", {}),
                "graph_expansions": _retrieval_trace(searcher).get("graph_expansions", []),
                "threshold_cutoff_reason": _retrieval_trace(searcher).get("threshold_cutoff_reason", ""),
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
) -> tuple[object, QuestionRunTrace, object]:
    planner_start = time.perf_counter()
    plan = plan_legal_query(question.question, planner_config)
    planner_ms = (time.perf_counter() - planner_start) * 1000.0
    retrieval_start = time.perf_counter()
    articles = _search_articles(searcher, question.question, top_k, plan=plan)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
    answer_start = time.perf_counter()
    evidence_blocks = build_evidence_blocks(articles)
    evidence_answer = generate_evidence_answer(question.question, plan, evidence_blocks, answer_config)
    answer_ms = (time.perf_counter() - answer_start) * 1000.0
    used_articles = articles_from_used_evidence(evidence_blocks, evidence_answer.used_evidence_ids)
    verification = verify_used_evidence_answer(
        evidence_answer.answer,
        used_articles,
        articles,
        question=question.question,
        required_components=plan.requested_components,
        target_norm_roles=plan.target_norm_roles,
        covered_components=evidence_answer.covered_components,
    )
    if evidence_answer.insufficient_evidence and not evidence_answer.used_evidence_ids:
        verification.issues.append("insufficient_evidence")
        verification.ok = False
    pred = format_prediction(question, evidence_answer.answer, used_articles)
    trace = QuestionRunTrace(
        id=question.id,
        question=question.question,
        predicted_metadata=plan.predicted_metadata(),
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
        answer_ms=round(answer_ms, 2),
        actual_backend=_backend_name(searcher),
    )
    return pred, trace, verification


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
    question_objects = [Question(id=int(item["id"]), question=str(item["question"])) for item in questions]
    predicted_metadata_by_id: dict[int, PredictedQuestionMetadata] = {}
    planner_predictions: list = []
    planner_traces: list[QuestionRunTrace] = []

    def planned_search(question: str, k: int):
        plan = planner_cache.get(question)
        if plan is None:
            plan = plan_legal_query(question, planner_config)
            planner_cache[question] = plan
        return _search_articles(hybrid, question, k, plan=plan)

    def planned_search_multihop(question: str, k: int):
        plan = planner_cache.get(question)
        if plan is None:
            plan = plan_legal_query(question, planner_config)
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
    if normalized == "hybrid_qdrant":
        hybrid_config = config_from_mapping(config)
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


def _search_articles(searcher: object, question: str, top_k: int, plan: LegalQueryPlan | None = None) -> list:
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


def _planner_config(config: dict, model: str | None, ollama_url: str, max_tokens: int) -> OllamaConfig:
    planning = config.get("planning", {})
    generation = config.get("generation", {})
    return OllamaConfig(
        model=model or planning.get("model") or generation.get("target_model") or DEFAULT_OLLAMA_MODEL,
        url=planning.get("ollama_url") or ollama_url or generation.get("ollama_url") or DEFAULT_OLLAMA_URL,
        max_tokens=int(planning.get("max_tokens", max_tokens)),
    )


def _answer_config(config: dict, model: str | None, ollama_url: str, max_tokens: int) -> OllamaConfig:
    generation = config.get("generation", {})
    return OllamaConfig(
        model=model or generation.get("target_model") or DEFAULT_OLLAMA_MODEL,
        url=generation.get("ollama_url") or ollama_url or DEFAULT_OLLAMA_URL,
        max_tokens=int(generation.get("max_tokens", max_tokens)),
    )


def _resolve_backend(cli_backend: str | None, config: dict) -> str:
    if cli_backend:
        return cli_backend
    configured = config.get("retrieval", {}).get("backend", "bm25_exact")
    if configured == "hybrid_qdrant" and not _hybrid_backend_ready(config):
        return "bm25_exact"
    return configured


def _resolve_articles_path(cli_articles: str, config: dict) -> str:
    return cli_articles or config.get("retrieval", {}).get("articles", "data/normalized/articles.jsonl")


def _resolve_index_path(cli_index: str | None, config: dict) -> str:
    return cli_index or config.get("retrieval", {}).get("bm25_index", "data/indices/bm25_index.json")


def _normalize_backend_name(backend: str) -> str:
    if backend in {"bm25", "bm25_baseline", "bm25_exact"}:
        return "bm25_exact"
    return backend


def _backend_name(searcher: object) -> str:
    if isinstance(searcher, HybridRetriever):
        return "hybrid_qdrant"
    if isinstance(searcher, BM25Index):
        return "bm25_exact"
    return type(searcher).__name__


def _hybrid_backend_ready(config: dict) -> bool:
    report_path = Path(config.get("retrieval", {}).get("hybrid_report", "data/indices/hybrid_index_report.json"))
    if report_path.exists():
        return True
    qdrant_path = Path(config.get("qdrant", {}).get("path", ""))
    if not qdrant_path.exists():
        return False
    files = [path for path in qdrant_path.rglob("*") if path.is_file()]
    return len(files) > 2


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
