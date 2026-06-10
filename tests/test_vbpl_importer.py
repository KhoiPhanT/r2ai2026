from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legal_rag.corpus.ingest import read_articles_jsonl
from legal_rag.corpus.vbpl import clean_generated_data, import_vbpl_corpus, inspect_vbpl_corpus


class VbplImporterTest(unittest.TestCase):
    def test_inspect_and_import_vbpl_corpus_to_canonical_articles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "law_data_raw"
            raw.mkdir()
            _write_jsonl(
                raw / "documents.jsonl",
                [
                    {
                        "document_id": "d1",
                        "document_number": "01/2020/QH14",
                        "document_type": "Luật",
                        "title": "Luật 01/2020/QH14 Luật X",
                        "summary": "Luật X",
                        "issuing_agency": "Quốc hội",
                        "issue_date": "2020-01-01T00:00:00",
                        "effective_date": "2020-07-01T00:00:00",
                        "expiration_date": "",
                        "legal_status": "Còn hiệu lực",
                        "language": "vi",
                        "source_url": "https://example.test/d1",
                        "html_url": "https://example.test/d1",
                        "article_count": 1,
                        "clause_count": 1,
                        "point_count": 1,
                        "chunk_count": 3,
                        "full_text": "Điều 1. Điều kiện hỗ trợ\n1. Doanh nghiệp nhỏ và vừa được hỗ trợ:\na) Có hồ sơ hợp lệ.",
                    },
                    {
                        "document_id": "d1b",
                        "document_number": "01/2020/QH14",
                        "document_type": "Luật",
                        "title": "Luật 01/2020/QH14 Luật X bản khác",
                        "legal_status": "Còn hiệu lực",
                        "language": "vi",
                        "article_count": 0,
                        "clause_count": 0,
                        "point_count": 0,
                        "full_text": "Điều 1. Bản khác",
                    },
                    {
                        "document_id": "eng",
                        "document_number": "02/2020/ND-CP",
                        "document_type": "Bản dịch văn bản",
                        "title": "Decree 02/2020/ND-CP",
                        "legal_status": "Còn hiệu lực",
                        "language": "vi",
                        "article_count": 0,
                        "full_text": "Article 1. This decree applies to enterprises.",
                    },
                    {
                        "document_id": "decision",
                        "document_number": "03/2020/QĐ-TTg",
                        "document_type": "Quyết định",
                        "title": "Quyết định 03/2020/QĐ-TTg",
                        "legal_status": "Còn hiệu lực",
                        "language": "vi",
                        "article_count": 1,
                        "full_text": "Điều 1. Nội dung quyết định.",
                    },
                ],
            )
            _write_jsonl(
                raw / "legal_units.jsonl",
                [
                    {
                        "unit_id": "d1__article_1",
                        "unit_type": "article",
                        "document_id": "d1",
                        "document_number": "01/2020/QH14",
                        "document_type": "Luật",
                        "document_title": "Luật 01/2020/QH14 Luật X",
                        "article_no": "1",
                        "article_title": "Điều kiện hỗ trợ",
                        "text": "Điều 1. Điều kiện hỗ trợ",
                    },
                    {
                        "unit_id": "d1__article_1__point_a",
                        "unit_type": "point",
                        "document_id": "d1",
                        "document_number": "01/2020/QH14",
                        "document_type": "Luật",
                        "document_title": "Luật 01/2020/QH14 Luật X",
                        "article_no": "1",
                        "article_title": "Điều kiện hỗ trợ",
                        "clause_no": "1",
                        "point_no": "a",
                        "text": "a) Có hồ sơ hợp lệ.",
                    },
                    {
                        "unit_id": "d1__article_1__clause_1",
                        "unit_type": "clause",
                        "document_id": "d1",
                        "document_number": "01/2020/QH14",
                        "document_type": "Luật",
                        "document_title": "Luật 01/2020/QH14 Luật X",
                        "article_no": "1",
                        "article_title": "Điều kiện hỗ trợ",
                        "clause_no": "1",
                        "text": "1. Doanh nghiệp nhỏ và vừa được hỗ trợ:",
                    },
                    {
                        "unit_id": "decision__article_1",
                        "unit_type": "article",
                        "document_id": "decision",
                        "document_number": "03/2020/QĐ-TTg",
                        "document_type": "Quyết định",
                        "document_title": "Quyết định 03/2020/QĐ-TTg",
                        "article_no": "1",
                        "article_title": "Nội dung",
                        "text": "Điều 1. Nội dung quyết định.",
                    },
                ],
            )

            inspect_report = inspect_vbpl_corpus(raw, root / "normalized" / "inspect.json")
            self.assertEqual(inspect_report.documents["valid_records"], 4)
            self.assertEqual(inspect_report.legal_units["valid_records"], 4)

            report = import_vbpl_corpus(raw, root / "normalized")

            self.assertEqual(report.selected_documents, 1)
            self.assertEqual(report.written_articles, 1)
            self.assertEqual(report.written_legal_units, 3)
            self.assertTrue(report.variant_conflicts)
            self.assertEqual(report.excluded_documents["excluded_doc_type:Bản dịch văn bản"], 1)
            self.assertEqual(report.excluded_documents["excluded_doc_type:Quyết định"], 1)

            articles = read_articles_jsonl(root / "normalized" / "articles.jsonl")
            self.assertEqual(len(articles), 1)
            article = articles[0]
            self.assertEqual(article.doc_id, "01/2020/QH14")
            self.assertEqual(article.article_label, "Điều 1")
            self.assertIn("Doanh nghiệp nhỏ và vừa", article.text)
            clauses = article.metadata["clause_nodes"]
            self.assertEqual(clauses[0]["label"], "Khoản 1")
            self.assertEqual(clauses[0]["point_nodes"][0]["label"], "Điểm a")

    def test_import_promotes_vietnamese_translation_records_without_cross_type_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "law_data_raw"
            raw.mkdir()
            _write_jsonl(
                raw / "documents.jsonl",
                [
                    {
                        "document_id": "law",
                        "document_number": "50/2005/QH11",
                        "document_type": "Bản dịch văn bản",
                        "title": "Luật 50/2005/QH11",
                        "legal_status": "Còn hiệu lực",
                        "language": "vi",
                        "article_count": 0,
                        "full_text": "Điều 100. Yêu cầu đối với đơn\nĐơn phải có tài liệu xác định khu vực địa lý.",
                    },
                    {
                        "document_id": "resolution",
                        "document_number": "50/2005/QH11",
                        "document_type": "Nghị quyết",
                        "title": "Nghị quyết 50/2005/QH11 về chương trình giám sát",
                        "legal_status": "Còn hiệu lực",
                        "language": "vi",
                        "article_count": 0,
                        "full_text": "Điều 1. Chương trình giám sát của Quốc hội.",
                    },
                    {
                        "document_id": "english_translation",
                        "document_number": "51/2005/QH11",
                        "document_type": "Bản dịch văn bản",
                        "title": "Luật 51/2005/QH11",
                        "legal_status": "Còn hiệu lực",
                        "language": "vi",
                        "article_count": 0,
                        "full_text": (
                            "Law on sample regulation. Article 1. This Law applies to organizations and individuals. "
                            "Article 2. The Government and the Ministry shall implement this Law according to its provisions."
                        ),
                    },
                ],
            )
            _write_jsonl(raw / "legal_units.jsonl", [])

            report = import_vbpl_corpus(raw, root / "normalized")
            articles = read_articles_jsonl(root / "normalized" / "articles.jsonl")

            self.assertEqual(report.selected_documents, 2)
            self.assertEqual(report.excluded_documents["english_like_or_translation"], 1)
            self.assertEqual({article.doc_type for article in articles}, {"Luật", "Nghị quyết"})
            self.assertEqual({article.doc_id for article in articles}, {"50/2005/QH11"})
            self.assertTrue(any(article.article_label == "Điều 100" for article in articles))

    def test_clean_generated_data_refuses_raw_source_and_deletes_only_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "law_data_raw").mkdir(parents=True)
            (root / "normalized").mkdir()
            (root / "indices").mkdir()

            removed = clean_generated_data(root)

            self.assertEqual({Path(path).name for path in removed}, {"normalized", "indices"})
            self.assertTrue((root / "law_data_raw").exists())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
