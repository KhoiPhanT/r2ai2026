# R2AI Legal RAG

Competition-oriented Vietnamese Legal RAG pipeline.

The current implementation is a local-first competition pipeline:

- corpus ingestion contract for official legal documents;
- article-level parser for `Dieu/Khoan/Diem`;
- mandatory corpus-grounded Ollama legal query planning before final retrieval;
- phrase-aware FTS/exact retrieval plus Qdrant dense+sparse retrieval and BGE reranking;
- micro-chunk indexing for long articles with parent promotion back to canonical articles;
- Ollama-backed evidence answer generation from direct support spans with `used_evidence_ids`;
- template-only retrieval debugging that is blocked from submission packaging;
- rules-first verifier over canonical used evidence;
- `results.json` validator and flat zip packager.

Final submissions are fail-closed: planner, retrieval, generator, used-evidence verifier, and package manifest must all pass.
Exact/phrase candidates reserve reranker capacity so strong legal wording is not displaced by broad semantic branches.
Adjacent and cross-reference hits may expand retrieval, but only direct evidence can support answer claims or appear in
`relevant_docs`/`relevant_articles`.

## Quick Commands

```bash
python3 -m pip install '.[rag]'
ollama pull qwen3:8b-q8_0
ollama show qwen3:8b-q8_0
python3 -m legal_rag.cli inspect_vbpl_corpus --input data/law_data_raw --report data/normalized/vbpl_inspect_report.json
python3 -m legal_rag.cli import_vbpl_corpus --input data/law_data_raw --output data/normalized
python3 -m legal_rag.cli build_index --input data/normalized/articles.jsonl --output data/indices/bm25_index.json
docker run -d --name r2ai-qdrant -p 6333:6333 -v "$PWD/data/indices/qdrant_server:/qdrant/storage" qdrant/qdrant:latest
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
`hybrid_qdrant`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, and a running Qdrant server at
`http://127.0.0.1:6333`. Embedded Qdrant is intentionally refused for large corpora because it times out on the
VBPL-scale collection.

Quick Ollama API check:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:8b-q8_0","messages":[{"role":"user","content":"Trả lời đúng một từ: OK"}],"temperature":0,"max_tokens":16,"reasoning_effort":"none"}'
```

## Corpus Record Contract

The default corpus path now expects the VBPL adapter source under `data/law_data_raw`:

- `documents.jsonl`
- `legal_units.jsonl`
- `metadata.csv`

Run `inspect_vbpl_corpus` first, then `import_vbpl_corpus`. The importer preserves raw source files and writes
canonical artifacts under `data/normalized`. It defaults to precision-first Vietnamese legal sources and excludes
translations, letters, directives, and most decisions unless explicitly enabled.

The importer derives the effective Vietnamese document type from a clearly typed Vietnamese title when the raw source
uses the generic `Bản dịch văn bản` label, but it detects language from the body and still excludes English translations.
Canonical dedupe is scoped by both document number and document type so a Law and a Resolution sharing a number do not
silently replace one another.

Legacy DOCX/JSON ingestion is still supported for smaller handoff corpora.

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
