# Runbook

## Corpus Handoff

For the current VBPL corpus, put these raw source files under `data/law_data_raw/` and keep them unchanged:

- `documents.jsonl`
- `legal_units.jsonl`
- `metadata.csv`

Inspect and import them into the canonical internal schema:

```bash
python3 -m legal_rag.cli inspect_vbpl_corpus \
  --input data/law_data_raw \
  --report data/normalized/vbpl_inspect_report.json

python3 -m legal_rag.cli import_vbpl_corpus \
  --input data/law_data_raw \
  --output data/normalized
```

Review `vbpl_import_report.json` before rebuilding indices. The importer:

- keeps raw files unchanged;
- infers an effective legal type only for clearly typed Vietnamese titles;
- detects English-like content from the document body, not the title/language flag alone;
- deduplicates within `(document_number, document_type)`, preserving different legal instruments that share a number.

To remove old generated artifacts before a fresh import, use:

```bash
python3 -m legal_rag.cli clean_generated_data --data-root data
```

This command only removes `data/law_data_normalized`, `data/normalized`, and `data/indices`; it refuses to delete
`data/law_data_raw`.

Legacy JSON/JSONL document records are still supported. They must include:

- `doc_id`
- `doc_type`
- `trich_yeu`
- `title_for_submission`
- `issuer`
- `issue_date`
- `effective_date`
- `status`
- `source_url`
- `raw_text`

`title_for_submission` must follow the competition format:

```text
Loai van ban + Ma van ban + Trich yeu
```

Example:

```text
Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa
```

## Submission Pipeline With Real Model

Install the strict under-14B local generator first:

```bash
python3 -m pip install '.[rag]'
ollama pull qwen3:8b-q8_0
ollama show qwen3:8b-q8_0
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:8b-q8_0","messages":[{"role":"user","content":"Trả lời đúng một từ: OK"}],"temperature":0,"max_tokens":16,"reasoning_effort":"none"}'
```

`configs/local_m4.json` uses a standalone Qdrant server at `http://127.0.0.1:6333`. For the VBPL-scale corpus,
do not use embedded Qdrant storage; the CLI refuses large embedded builds because local mode is too slow for this
collection size. Start Qdrant before `build_hybrid_index`:

```bash
docker rm -f r2ai-qdrant 2>/dev/null || true
docker run -d --name r2ai-qdrant \
  -p 6333:6333 \
  -v "$PWD/data/indices/qdrant_server:/qdrant/storage" \
  qdrant/qdrant:latest
curl http://127.0.0.1:6333/collections
```

`run_batch` must use Ollama twice per final question: first as a legal query planner, then as an
evidence answerer. It fails instead of silently falling back when either step is unavailable or returns
invalid JSON.

The production retrieval path applies these gates in order:

1. FTS corpus resolver supplies real document candidates to the planner.
2. Lexicon candidates must be grounded by phrase or Vietnamese multi-token overlap with the question.
3. Exact and top phrase hits reserve capacity in the reranker candidate set.
4. BGE reranks query-focused support spans, using 640 tokens for simple cases and 1024 for complex cases.
5. Ranking combines normalized reranker/fusion/structured scores at `0.70/0.20/0.10`.
6. Adjacent/cross-reference evidence can drive traversal but is removed before answer generation.
7. Claims, citations, and final relevant articles are reconstructed only from direct used evidence.

Ollama context is adaptive: ordinary planner/answer calls use `8192/12288`; only comparison or genuinely
multi-component hard cases raise context and enable thinking. Narrow component repair is deterministic first and uses
a non-thinking constrained rewrite only when direct extraction cannot close the gap.

The active collection is `r2ai_law_articles_v2`. Its point IDs include both the canonical span key and encoded text
hash, preserving distinct repeated clause labels instead of silently overwriting them in Qdrant.

Batch progress is tied to a fingerprint of questions, corpus/index artifacts, config, and all `legal_rag/*.py` source.
After code changes, use a new `--output` path; the CLI refuses to mix old and new pipeline behavior.

```bash
python3 -m legal_rag.cli inspect_vbpl_corpus \
  --input data/law_data_raw \
  --report data/normalized/vbpl_inspect_report.json

python3 -m legal_rag.cli import_vbpl_corpus \
  --input data/law_data_raw \
  --output data/normalized

python3 -m legal_rag.cli build_index \
  --input data/normalized/articles.jsonl \
  --output data/indices/bm25_index.json

docker run -d --name r2ai-qdrant \
  -p 6333:6333 \
  -v "$PWD/data/indices/qdrant_server:/qdrant/storage" \
  qdrant/qdrant:latest

python3 -m legal_rag.cli build_hybrid_index \
  --input data/normalized/articles.jsonl \
  --config configs/local_m4.json

python3 -m legal_rag.cli plan_query \
  --question "Luật Thủ đô quy định những chính sách đặc thù nào?" \
  --config configs/local_m4.json

python3 -m legal_rag.cli debug_pipeline \
  --question "Luật Thủ đô quy định những chính sách đặc thù nào?" \
  --config configs/local_m4.json

python3 -m legal_rag.cli run_batch \
  --questions data/test.json \
  --index data/indices/bm25_index.json \
  --output data/submissions/results.json \
  --config configs/local_m4.json \
  --model qwen3:8b-q8_0

python3 -m legal_rag.cli validate_submission \
  --input data/submissions/results.json \
  --questions data/test.json

python3 -m legal_rag.cli package_submission \
  --input data/submissions/results.json \
  --output data/submissions/submission.zip
```

Scaffold an internal tune/holdout annotation split directly from the organizer-style questions:

```bash
python3 -m legal_rag.cli prepare_gold_metadata \
  --questions data/test.json \
  --output-dir data/questions \
  --config configs/local_m4.json
```

This writes `data/questions/dev_gold.jsonl`, `data/questions/holdout_gold.jsonl`, and a split manifest.
The files are scaffolds: predicted metadata is prefilled, but `gold_relevant_docs` and `gold_relevant_articles`
must still be reviewed and completed by hand.

When a labeled dev set is available, compare retrieval and metadata-aware pipeline metrics:

```bash
python3 -m legal_rag.cli eval_pipeline \
  --questions data/test.json \
  --expected data/questions/dev_gold.jsonl \
  --config configs/local_m4.json
```

`validate_submission` checks JSON shape only. `package_submission` requires the adjacent
`results.manifest.json` to show `planner_backend: ollama`, `generator_backend: ollama`, and no verifier issues.
`relevant_docs` and `relevant_articles` are derived only from the answerer's `used_evidence_ids`.

## Debug Retrieval

This command uses the old template baseline only to inspect retrieval. Its output is marked
`generator_backend: template_debug` and cannot be packaged.

```bash
python3 -m legal_rag.cli debug_retrieval \
  --question "Luật Thủ đô quy định những chính sách đặc thù nào?" \
  --index data/indices/bm25_index.json \
  --backend hybrid_qdrant \
  --config configs/local_m4.json
```

Use `--backend bm25_exact` to compare against the lexical baseline.

## Debug Full Pipeline

Use this when you need to see whether the system is actually planning legal queries rather than searching
with the raw question only.

```bash
python3 -m legal_rag.cli debug_pipeline \
  --question "Luật Thủ đô quy định những chính sách đặc thù nào?" \
  --index data/indices/bm25_index.json \
  --backend hybrid_qdrant \
  --config configs/local_m4.json
```

The output includes planner JSON, planned queries, retrieval traces, canonical articles, support snippets,
and evidence ids.

## Smoke Test

```bash
python3 -m legal_rag.cli inspect_vbpl_corpus --input data/law_data_raw --report /private/tmp/r2ai-smoke/vbpl_inspect_report.json
python3 -m legal_rag.cli import_vbpl_corpus --input data/law_data_raw --output /private/tmp/r2ai-smoke
python3 -m legal_rag.cli build_index --input /private/tmp/r2ai-smoke/articles.jsonl --output /private/tmp/r2ai-smoke/bm25_index.json
python3 -m legal_rag.cli build_hybrid_index --input /private/tmp/r2ai-smoke/articles.jsonl --config configs/local_m4.json
python3 -m legal_rag.cli plan_query --question "Luật Thủ đô quy định những chính sách đặc thù nào?" --config configs/local_m4.json
python3 -m legal_rag.cli debug_pipeline --question "Luật Thủ đô quy định những chính sách đặc thù nào?" --index /private/tmp/r2ai-smoke/bm25_index.json --articles /private/tmp/r2ai-smoke/articles.jsonl --config configs/local_m4.json
python3 -m legal_rag.cli ask --question "Luật Thủ đô quy định những chính sách đặc thù nào?" --index /private/tmp/r2ai-smoke/bm25_index.json --articles /private/tmp/r2ai-smoke/articles.jsonl --config configs/local_m4.json --model qwen3:8b-q8_0
python3 -m legal_rag.cli run_batch --questions tests/fixtures/sample_questions.json --index /private/tmp/r2ai-smoke/bm25_index.json --articles /private/tmp/r2ai-smoke/articles.jsonl --output /private/tmp/r2ai-smoke/results.json --config configs/local_m4.json --model qwen3:8b-q8_0
python3 -m legal_rag.cli validate_submission --input /private/tmp/r2ai-smoke/results.json --questions tests/fixtures/sample_questions.json
python3 -m legal_rag.cli package_submission --input /private/tmp/r2ai-smoke/results.json --output /private/tmp/r2ai-smoke/submission.zip
```

## Offline Model Mode

Set `embedding.local_files_only=true` and `reranker.local_files_only=true` in `configs/local_m4.json` after the
BGE models are cached locally. In that mode query-time model loading fails clearly instead of reaching out to
Hugging Face.
