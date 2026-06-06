# R2AI Legal RAG

Competition-oriented Vietnamese Legal RAG pipeline.

The current implementation is a local-first competition pipeline:

- corpus ingestion contract for official legal documents;
- article-level parser for `Dieu/Khoan/Diem`;
- mandatory Ollama legal query planning before final retrieval;
- BM25/exact retrieval baseline plus Qdrant hybrid retrieval with BGE-M3 and BGE reranking;
- micro-chunk indexing for long articles with parent promotion back to canonical articles;
- Ollama-backed evidence answer generation with `used_evidence_ids`;
- template-only retrieval debugging that is blocked from submission packaging;
- rules-first verifier over canonical used evidence;
- `results.json` validator and flat zip packager.

Final submissions are fail-closed: planner, retrieval, generator, used-evidence verifier, and package manifest must all pass.

## Quick Commands

```bash
python3 -m pip install '.[rag]'
ollama pull qwen3:8b-q8_0
ollama show qwen3:8b-q8_0
python3 -m legal_rag.cli normalize_docs --input data/law_data_raw --output data/law_data_normalized
python3 -m legal_rag.cli ingest_corpus --input data/law_data_normalized/documents.jsonl --output data/normalized/articles.jsonl
python3 -m legal_rag.cli build_index --input data/normalized/articles.jsonl --output data/indices/bm25_index.json
python3 -m legal_rag.cli build_hybrid_index --input data/normalized/articles.jsonl --config configs/local_m4.json
python3 -m legal_rag.cli plan_query --question "Luật Thủ đô quy định những chính sách đặc thù nào?" --config configs/local_m4.json
python3 -m legal_rag.cli debug_pipeline --question "Luật Thủ đô quy định những chính sách đặc thù nào?" --config configs/local_m4.json
python3 -m legal_rag.cli ask --question "Luật Thủ đô quy định những chính sách đặc thù nào?" --index data/indices/bm25_index.json --config configs/local_m4.json --model qwen3:8b-q8_0
python3 -m legal_rag.cli run_batch --questions data/test.json --index data/indices/bm25_index.json --output results.json --config configs/local_m4.json --model qwen3:8b-q8_0
python3 -m legal_rag.cli eval_pipeline --questions data/questions/dev.json --expected data/questions/dev_expected.json --config configs/local_m4.json
python3 -m legal_rag.cli validate_submission --input results.json --questions data/test.json
python3 -m legal_rag.cli package_submission --input results.json --output submission.zip
```

`validate_submission` checks file shape only. `package_submission` additionally requires a `results.manifest.json`
showing `planner_backend: ollama`, `generator_backend: ollama`, and no verifier issues. Use `debug_retrieval`
only to inspect the old template retrieval baseline; debug output is intentionally refused by `package_submission`.
Use `debug_pipeline` to inspect planner JSON, planned queries, reranked evidence, and evidence ids.

For BM25-only debugging, pass `--backend bm25_exact`. For the strong path, `configs/local_m4.json` selects
`hybrid_qdrant`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, and Qdrant embedded storage at
`data/indices/qdrant`. Remove `qdrant.path` from the config if you prefer a running Qdrant server at `qdrant.url`.

Quick Ollama API check:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:8b-q8_0","messages":[{"role":"user","content":"Trả lời đúng một từ: OK"}],"temperature":0,"max_tokens":16,"reasoning_effort":"none"}'
```

## Corpus Record Contract

Each official legal document should be supplied as JSON/JSONL with:

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

`title_for_submission` must follow the competition format: `Loai van ban + Ma van ban + Trich yeu`.
