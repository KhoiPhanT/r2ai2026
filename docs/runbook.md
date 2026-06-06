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

## Baseline Pipeline

```bash
python3 -m legal_rag.cli ingest_corpus \
  --input data/raw \
  --output data/normalized/articles.jsonl

python3 -m legal_rag.cli build_index \
  --input data/normalized/articles.jsonl \
  --output data/indices/bm25_index.json

python3 -m legal_rag.cli run_batch \
  --questions data/test.json \
  --index data/indices/bm25_index.json \
  --output data/submissions/results.json

python3 -m legal_rag.cli validate_submission \
  --input data/submissions/results.json \
  --questions data/test.json

python3 -m legal_rag.cli package_submission \
  --input data/submissions/results.json \
  --output data/submissions/submission.zip
```

## Smoke Test

```bash
python3 -m legal_rag.cli ingest_corpus --input data/raw/sample --output /private/tmp/r2ai-smoke/articles.jsonl
python3 -m legal_rag.cli build_index --input /private/tmp/r2ai-smoke/articles.jsonl --output /private/tmp/r2ai-smoke/bm25_index.json
python3 -m legal_rag.cli run_batch --questions tests/fixtures/sample_questions.json --index /private/tmp/r2ai-smoke/bm25_index.json --output /private/tmp/r2ai-smoke/results.json
python3 -m legal_rag.cli validate_submission --input /private/tmp/r2ai-smoke/results.json --questions tests/fixtures/sample_questions.json
python3 -m legal_rag.cli package_submission --input /private/tmp/r2ai-smoke/results.json --output /private/tmp/r2ai-smoke/submission.zip
```

## Next Integration Step

After corpus coverage is acceptable, add Qdrant + BGE-M3 as a second retriever branch while keeping this BM25/exact baseline as the regression floor.

