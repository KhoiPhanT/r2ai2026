from __future__ import annotations

import unittest

from legal_rag.parser.legal_text import parse_document
from legal_rag.schemas.models import DocumentRecord


class LegalParserTest(unittest.TestCase):
    def test_relation_extraction_is_sentence_local_and_precision_first(self) -> None:
        record = DocumentRecord(
            doc_id="02/2020/NĐ-CP",
            doc_type="Nghị định",
            trich_yeu="Quy định chi tiết Luật X",
            title_for_submission="Nghị định 02/2020/NĐ-CP Quy định chi tiết Luật X",
            raw_text=(
                "Nghị định này quy định chi tiết Luật số 03/2020/QH14. "
                "Căn cứ Luật số 01/2020/QH14; Căn cứ Luật số 04/2020/QH14.\n\n"
                "Điều 1. Hồ sơ\nHồ sơ gồm đơn đề nghị."
            ),
            metadata={
                "header_text": "Căn cứ Luật số 01/2020/QH14; Căn cứ Luật số 04/2020/QH14.",
            },
        )

        articles = parse_document(record)
        relations = articles[0].metadata["document_relations"]

        self.assertEqual(relations["guides_doc_ids"], ["03/2020/QH14"])
        self.assertEqual(relations["amends_doc_ids"], [])
        reference_doc_ids = {row["target_doc_id"] for row in relations["references_only"] if row.get("target_doc_id")}
        self.assertTrue({"01/2020/QH14", "04/2020/QH14"} <= reference_doc_ids)


if __name__ == "__main__":
    unittest.main()
