from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from legal_rag.corpus.ingest import ingest_corpus, read_articles_jsonl, write_articles_jsonl
from legal_rag.documents import normalize_documents
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
    OllamaConfig,
    OllamaError,
    generate_grounded_answer,
    generate_ollama_answer,
)
from legal_rag.retrieval import (
    BM25Index,
    HybridRetrievalError,
    HybridRetriever,
    build_hybrid_index,
    config_from_mapping,
    evaluate_retrieval,
    retrieve_articles,
)
from legal_rag.schemas.models import Question
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

    hybrid_build = subparsers.add_parser("build_hybrid_index")
    hybrid_build.add_argument("--input", required=True)
    hybrid_build.add_argument("--config", default="configs/local_m4.json")
    hybrid_build.add_argument("--report", default="data/indices/hybrid_index_report.json")

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

    eval_cmd = subparsers.add_parser("eval_retrieval")
    eval_cmd.add_argument("--questions", required=True)
    eval_cmd.add_argument("--expected", required=True)
    eval_cmd.add_argument("--index", default="data/indices/bm25_index.json")
    eval_cmd.add_argument("--articles")
    eval_cmd.add_argument("--backend")
    eval_cmd.add_argument("--config", default="configs/local_m4.json")
    eval_cmd.add_argument("--top-k", type=int, default=5)

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
    if args.command == "normalize_docs":
        return _cmd_normalize_docs(args.input, args.output)
    if args.command == "build_index":
        return _cmd_build_index(args.input, args.output)
    if args.command == "build_hybrid_index":
        return _cmd_build_hybrid_index(args.input, args.config, args.report)
    if args.command == "run_batch":
        config = _load_config(args.config)
        return _cmd_run_batch(
            args.questions,
            _resolve_index_path(args.index, config),
            args.output,
            args.top_k,
            OllamaConfig(model=args.model, url=args.ollama_url, max_tokens=args.max_tokens),
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
            OllamaConfig(model=args.model, url=args.ollama_url, max_tokens=args.max_tokens),
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
        report = build_hybrid_index(input_path, config_from_mapping(config))
    except HybridRetrievalError as exc:
        print(f"hybrid index build failed: {exc}")
        return 1
    _write_report(Path(report_path), report.to_dict())
    print(f"indexed {report.articles} articles to qdrant collection {report.collection}")
    print(f"report: {report_path}")
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
    batch_start = time.time()

    # --- Stream: process remaining questions one by one ---
    with progress_path.open("a", encoding="utf-8") as progress_file:
        for step, question in enumerate(remaining, start=1):
            q_start = time.time()
            done_count = initial_done + step
            articles = _search_articles(searcher, question.question, top_k)
            try:
                answer = generate_ollama_answer(question.question, articles, ollama_config)
            except OllamaError as exc:
                elapsed = time.time() - batch_start
                print(f"\n[{done_count}/{total}] FAIL id={question.id}: {exc} ({elapsed:.0f}s elapsed)")
                print("refusing to write submission output without a real model answer")
                print(f"progress saved: {progress_path} ({initial_done + step - 1} done)")
                return 1

            verification = verify_prediction_evidence(answer, articles)
            if not verification.ok:
                verifier_issues.append({"id": question.id, "issues": verification.issues})
                if not allow_verifier_issues:
                    print(f"\n[{done_count}/{total}] verifier failed id={question.id}: {verification.issues}")
                    print("refusing to write final submission output with verifier issues")
                    print(f"progress saved: {progress_path} ({initial_done + step - 1} done)")
                    return 1

            pred = format_prediction(question, answer, articles)
            pred_dict = pred.to_dict()

            # Append to progress file immediately
            progress_file.write(json.dumps(pred_dict, ensure_ascii=False) + "\n")
            progress_file.flush()

            done_by_id[question.id] = pred_dict

            # Streaming log
            q_elapsed = time.time() - q_start
            total_elapsed = time.time() - batch_start
            v_flag = " ⚠" if not verification.ok else " ✓"
            n_arts = len(articles)
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
            **retrieval_manifest,
            "generator_backend": "ollama",
            "model": ollama_config.model,
            "ollama_url": ollama_config.url,
            "verifier_issues": verifier_issues[:200],
            "allow_verifier_issues": allow_verifier_issues,
        },
    )
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
    articles = _search_articles(searcher, question, top_k)
    try:
        answer = generate_ollama_answer(question, articles, ollama_config)
    except OllamaError as exc:
        print(f"model generation failed: {exc}")
        return 1

    verification = verify_prediction_evidence(answer, articles)
    print(answer)
    print("\nCăn cứ được truy hồi:")
    for article in articles:
        print(f"- {article.relevant_article} (score={article.score:.4f})")
    if not verification.ok:
        print(f"\nverifier issues: {verification.issues}")
        return 1
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
        return BM25Index.load(index_path), {"retrieval_backend": "bm25_exact"}
    if normalized == "hybrid_qdrant":
        hybrid_config = config_from_mapping(config)
        retriever = HybridRetriever.load(articles_path, index_path, hybrid_config)
        return (
            retriever,
            {
                "retrieval_backend": "hybrid_qdrant",
                "qdrant_url": hybrid_config.qdrant_url,
                "qdrant_path": hybrid_config.qdrant_path,
                "qdrant_collection": hybrid_config.collection,
                "embedding_model": hybrid_config.embedding_model,
                "reranker_model": hybrid_config.reranker_model,
                "bm25_top_k": hybrid_config.bm25_top_k,
                "vector_top_k": hybrid_config.vector_top_k,
                "rerank_top_k": hybrid_config.rerank_top_k,
            },
        )
    raise HybridRetrievalError(f"unknown_retrieval_backend:{backend}")


def _search_articles(searcher: object, question: str, top_k: int) -> list:
    if isinstance(searcher, BM25Index):
        return retrieve_articles(searcher, question, top_k=top_k)
    if isinstance(searcher, HybridRetriever):
        return searcher.search(question, top_k=top_k)
    raise HybridRetrievalError(f"unsupported_searcher:{type(searcher).__name__}")


def _resolve_backend(cli_backend: str | None, config: dict) -> str:
    return cli_backend or config.get("retrieval", {}).get("backend", "bm25_exact")


def _resolve_articles_path(cli_articles: str, config: dict) -> str:
    return cli_articles or config.get("retrieval", {}).get("articles", "data/normalized/articles.jsonl")


def _resolve_index_path(cli_index: str | None, config: dict) -> str:
    return cli_index or config.get("retrieval", {}).get("bm25_index", "data/indices/bm25_index.json")


def _normalize_backend_name(backend: str) -> str:
    if backend in {"bm25", "bm25_baseline", "bm25_exact"}:
        return "bm25_exact"
    return backend


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

    ollama_config = OllamaConfig(model=model, url=ollama_url, max_tokens=max_tokens)
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
