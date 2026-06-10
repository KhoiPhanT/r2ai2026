from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from legal_rag.corpus.ingest import write_articles_jsonl
from legal_rag.lexicon import build_legal_lexicon, load_legal_lexicon, search_legal_lexicon
from legal_rag.retrieval.fts import FTS5Index, build_fts5_index
from legal_rag.schemas.models import ArticleNode


class LexicalIndexTest(unittest.TestCase):
    def test_fts_search_and_exact_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            articles = [
                _article(
                    "80/2021/NĐ-CP",
                    "Điều 22",
                    "Hỗ trợ doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo",
                    "Hỗ trợ chi phí thuê mặt bằng. Thời gian hỗ trợ tối đa là 03 năm.",
                ),
                _article(
                    "94/2009/NĐ-CP",
                    "Điều 12",
                    "Quản lý cai nghiện",
                    "Hỗ trợ người cai nghiện ma túy bắt buộc.",
                ),
            ]
            articles_path = root / "articles.jsonl"
            index_path = root / "articles.sqlite"
            write_articles_jsonl(articles, articles_path)
            build_fts5_index(articles_path, index_path)

            index = FTS5Index.load(index_path)
            hits = index.search("doanh nghiệp nhỏ và vừa thuê mặt bằng thời gian hỗ trợ", top_k=2)
            exact = index.exact_search("80/2021/NĐ-CP Điều 22", top_k=2)

            self.assertEqual(hits[0].doc_id, "80/2021/NĐ-CP")
            self.assertEqual(exact[0].relevant_article, articles[0].relevant_article)
            index.close()

    def test_sqlite_lexicon_preserves_sme_alias_without_cross_domain_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            articles = [
                _article(
                    "80/2021/NĐ-CP",
                    "Điều 22",
                    "Hỗ trợ doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo",
                    "Hỗ trợ chi phí thuê mặt bằng. Thời gian hỗ trợ tối đa là 03 năm.",
                ),
                _article(
                    "94/2009/NĐ-CP",
                    "Điều 12",
                    "Chính sách hỗ trợ cai nghiện",
                    "Hỗ trợ đối tượng cai nghiện ma túy bắt buộc.",
                ),
            ]
            articles_path = root / "articles.jsonl"
            lexicon_path = root / "lexicon.sqlite"
            write_articles_jsonl(articles, articles_path)
            build_legal_lexicon(articles_path, lexicon_path)

            lexicon = load_legal_lexicon(lexicon_path)
            matches = search_legal_lexicon(
                lexicon,
                "Công ty nhỏ và vừa được hỗ trợ giá thuê mặt bằng trong bao lâu?",
                limit=4,
            )
            combined = " ".join(match.term for match in matches).lower()

            self.assertIn("doanh nghiệp nhỏ và vừa", combined)
            self.assertIn("thuê mặt bằng", combined)
            self.assertNotIn("cai nghiện", combined)
            lexicon.close()

    def test_explicit_expiration_relation_marks_old_document_superseded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = _article(
                "12/2022/NĐ-CP",
                "Điều 62",
                "Hiệu lực thi hành",
                "Nghị định số 28/2020/NĐ-CP hết hiệu lực thi hành kể từ ngày Nghị định này có hiệu lực.",
            )
            old = _article(
                "28/2020/NĐ-CP",
                "Điều 8",
                "Giao kết hợp đồng lao động",
                "Phạt tiền đối với hành vi giữ bản chính văn bằng của người lao động.",
            )
            articles_path = root / "articles.jsonl"
            index_path = root / "articles.sqlite"
            write_articles_jsonl([current, old], articles_path)
            build_fts5_index(articles_path, index_path)

            index = FTS5Index.load(index_path)
            try:
                superseded = index.superseded_doc_ids({current.doc_id, old.doc_id})
                sources = index.superseding_sources({old.doc_id})
            finally:
                index.close()

            self.assertEqual(superseded, {"28/2020/NĐ-CP"})
            self.assertEqual(sources, {"28/2020/NĐ-CP": {"12/2022/NĐ-CP"}})


def _article(doc_id: str, article_label: str, article_title: str, text: str) -> ArticleNode:
    title = f"Nghị định số {doc_id} {article_title}"
    return ArticleNode(
        article_key=f"{doc_id}|{title}|{article_label}",
        doc_id=doc_id,
        doc_type="Nghị định",
        title_for_submission=title,
        article_label=article_label,
        article_title=article_title,
        text=text,
        metadata={"norm_roles": ["support_policy"]},
    )


if __name__ == "__main__":
    unittest.main()
