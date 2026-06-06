# Runbook

## Corpus Handoff

Put official legal documents under `data/raw/`. JSON and JSONL records must include:

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

`configs/local_m4.json` uses Qdrant embedded storage at `data/indices/qdrant` so Docker is not required.
Remove `qdrant.path` if you want to use a standalone Qdrant server at `qdrant.url`.

`run_batch` must use Ollama. It fails instead of silently falling back when the model is unavailable.

```bash
python3 -m legal_rag.cli normalize_docs \
  --input data/law_data_raw \
  --output data/law_data_normalized

python3 -m legal_rag.cli ingest_corpus \
  --input data/law_data_normalized/documents.jsonl \
  --output data/normalized/articles.jsonl

python3 -m legal_rag.cli build_index \
  --input data/normalized/articles.jsonl \
  --output data/indices/bm25_index.json

python3 -m legal_rag.cli build_hybrid_index \
  --input data/normalized/articles.jsonl \
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

`validate_submission` checks JSON shape only. `package_submission` requires the adjacent
`results.manifest.json` to show `generator_backend: ollama` and no verifier issues.

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

## Smoke Test

```bash
python3 -m legal_rag.cli normalize_docs --input data/law_data_raw --output data/law_data_normalized
python3 -m legal_rag.cli ingest_corpus --input data/law_data_normalized/documents.jsonl --output /private/tmp/r2ai-smoke/articles.jsonl
python3 -m legal_rag.cli build_index --input /private/tmp/r2ai-smoke/articles.jsonl --output /private/tmp/r2ai-smoke/bm25_index.json
python3 -m legal_rag.cli build_hybrid_index --input /private/tmp/r2ai-smoke/articles.jsonl --config configs/local_m4.json
python3 -m legal_rag.cli ask --question "Luật Thủ đô quy định những chính sách đặc thù nào?" --index /private/tmp/r2ai-smoke/bm25_index.json --articles /private/tmp/r2ai-smoke/articles.jsonl --config configs/local_m4.json --model qwen3:8b-q8_0
python3 -m legal_rag.cli run_batch --questions tests/fixtures/sample_questions.json --index /private/tmp/r2ai-smoke/bm25_index.json --articles /private/tmp/r2ai-smoke/articles.jsonl --output /private/tmp/r2ai-smoke/results.json --config configs/local_m4.json --model qwen3:8b-q8_0
python3 -m legal_rag.cli validate_submission --input /private/tmp/r2ai-smoke/results.json --questions tests/fixtures/sample_questions.json
python3 -m legal_rag.cli package_submission --input /private/tmp/r2ai-smoke/results.json --output /private/tmp/r2ai-smoke/submission.zip
```

## Next Integration Step

After corpus coverage is acceptable, add Qdrant + BGE-M3 as a second retriever branch while keeping this BM25/exact baseline as the regression floor.
