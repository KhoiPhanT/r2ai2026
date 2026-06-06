# R2AI Legal RAG

Competition-oriented Vietnamese Legal RAG pipeline.

The current implementation is an offline, dependency-light MVP:

- corpus ingestion contract for official legal documents;
- article-level parser for `Dieu/Khoan/Diem`;
- BM25 + exact-match retrieval baseline;
- answer templating constrained to retrieved evidence;
- rules-first verifier;
- `results.json` validator and flat zip packager.

The interfaces are intentionally shaped so Qdrant, LlamaIndex, BGE-M3, and BGE reranker can be plugged in after the corpus contract is stable.

## Quick Commands

```bash
python3 -m legal_rag.cli normalize_docs --input data/law_data_raw --output data/law_data_normalized
python3 -m legal_rag.cli ingest_corpus --input data/law_data_normalized/documents.jsonl --output data/normalized/articles.jsonl
python3 -m legal_rag.cli build_index --input data/normalized/articles.jsonl --output data/indices/bm25_index.json
python3 -m legal_rag.cli run_batch --questions data/test.json --index data/indices/bm25_index.json --output results.json
python3 -m legal_rag.cli validate_submission --input results.json --questions data/test.json
python3 -m legal_rag.cli package_submission --input results.json --output submission.zip
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
