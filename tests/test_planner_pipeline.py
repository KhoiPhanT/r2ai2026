from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from legal_rag.corpus.ingest import ingest_corpus, write_articles_jsonl
from legal_rag.cli import (
    _insufficient_answer_for_corpus_gap,
    _repair_citations_to_used_evidence,
    _repair_list_answer_from_used_evidence,
    _repair_missing_components_from_direct_evidence,
    _select_evidence_candidates,
)
from legal_rag.generation.evidence import (
    _answer_user_prompt,
    answer_runtime_config,
    articles_from_used_evidence,
    build_evidence_blocks,
    evidence_answer_from_json,
    finalize_evidence_answer,
)
from legal_rag.generation.ollama import OllamaConfig, OllamaError
from legal_rag.lexicon import LegalLexiconMatch, build_legal_lexicon, load_legal_lexicon, search_legal_lexicon
from legal_rag.planner import LegalQueryPlan, PlannedQuery, legal_query_plan_from_json, planner_runtime_config
from legal_rag.question_metadata import infer_governing_doc_hints, infer_runtime_metadata
from legal_rag.retrieval import BM25Index
from legal_rag.retrieval.hybrid import (
    HybridRetrievalConfig,
    HybridRetriever,
    build_hybrid_index,
    _apply_retrieval_bias,
    _attach_plan_support_span,
    _promote_and_dedupe,
    _suppress_unrequested_local_scope,
)
from legal_rag.schemas.models import ArticleNode, CorpusCandidate
from legal_rag.verifier import verify_used_evidence_answer


class FakeEmbedder:
    def encode(self, texts, return_dense=True, return_sparse=True, return_colbert_vecs=False):
        return {
            "dense_vecs": [[float(i + 1)] * 1024 for i, _text in enumerate(texts)],
            "lexical_weights": [{str(i + 1): 1.0} for i, _text in enumerate(texts)],
        }


class FakeReranker:
    def compute_score(self, pairs, normalize=True):
        return [1.0 - i * 0.01 for i, _pair in enumerate(pairs)]


class CountingQdrantClient:
    def __init__(self, points=None):
        self.points = points or []
        self.queries = []
        self.upserted = []

    def collection_exists(self, collection):
        return False

    def delete_collection(self, collection):
        raise AssertionError("unexpected delete")

    def create_collection(self, collection, vectors_config, sparse_vectors_config):
        return None

    def upsert(self, collection_name, points):
        self.upserted.extend(points)

    def query_points(self, collection_name, query, using, with_payload, limit):
        self.queries.append(using)
        return type("Result", (), {"points": self.points[:limit]})()


class PlannerPipelineTest(unittest.TestCase):
    def test_planner_drops_lexicon_candidate_with_only_generic_token_overlap(self) -> None:
        candidate = LegalLexiconMatch(
            term="chuẩn hóa dữ liệu đăng ký doanh nghiệp",
            aliases=[],
            related_terms=[],
            doc_ids=["168/2025/NĐ-CP"],
            article_keys=[],
            norm_roles=["procedure"],
            doc_types=["Nghị định"],
            regimes=["Nghị định về đăng ký doanh nghiệp"],
            source_spans=[],
            specificity=0.8,
            score=4.5,
        )

        plan = legal_query_plan_from_json(
            {
                "intent": "procedure",
                "normalized_question": "nộp đơn đăng ký chỉ dẫn địa lý cần tài liệu gì",
                "question_type": "list",
                "answer_shape": "list_items",
                "requested_components": ["hồ sơ"],
                "entities": {"subjects": [], "actions": [], "objects": [], "conditions": [], "amounts_or_deadlines": []},
                "target_doc_ids": [],
                "target_article_labels": [],
                "queries": [],
            },
            "Khi nộp đơn đăng ký chỉ dẫn địa lý, công ty cần chuẩn bị tài liệu gì?",
            lexicon_candidates=[candidate],
        )

        self.assertNotIn(candidate.term, plan.lexical_expansions)
        self.assertFalse(any("168/2025" in item for item in plan.governing_doc_hints))

    def test_planner_reasoning_is_adaptive_not_global(self) -> None:
        config = OllamaConfig(think=False, num_ctx=8192, max_tokens=300)

        simple = planner_runtime_config("Thời hạn giải quyết là bao lâu?", config)
        complex_case = planner_runtime_config("So sánh điều kiện và thủ tục áp dụng của hai chính sách này?", config)

        self.assertFalse(simple.think)
        self.assertEqual(simple.num_ctx, 8192)
        self.assertTrue(complex_case.think)
        self.assertEqual(complex_case.num_ctx, 12288)

    def test_answer_reasoning_escalates_only_for_hard_initial_case(self) -> None:
        config = OllamaConfig(think=False, num_ctx=12288, max_tokens=520)
        two_components = LegalQueryPlan(
            intent="penalty",
            question_scope="unknown",
            normalized_question="xử phạt và khắc phục",
            requested_components=["mức phạt", "biện pháp khắc phục"],
        )
        comparison = LegalQueryPlan(
            intent="comparison",
            question_scope="multi_doc",
            normalized_question="so sánh hai quy định",
            question_type="comparison",
        )

        self.assertFalse(answer_runtime_config(two_components, config).think)
        self.assertEqual(answer_runtime_config(two_components, config).max_tokens, 360)
        self.assertTrue(answer_runtime_config(comparison, config).think)

    def test_retriever_suppresses_explicitly_superseded_candidate(self) -> None:
        class LexicalWithStatus:
            def superseded_doc_ids(self, candidate_doc_ids):
                self.candidates = candidate_doc_ids
                return {"28/2020/NĐ-CP"}

        current = self._article("Điều 9", doc_id="12/2022/NĐ-CP")
        old = self._article("Điều 8", doc_id="28/2020/NĐ-CP")
        lexical = LexicalWithStatus()
        retriever = HybridRetriever([], None, HybridRetrievalConfig(), lexical_index=lexical)

        kept, removed = retriever._suppress_superseded_candidates([current, old])

        self.assertEqual([article.doc_id for article in kept], ["12/2022/NĐ-CP"])
        self.assertEqual(removed, {"28/2020/NĐ-CP"})
        self.assertEqual(lexical.candidates, {"12/2022/NĐ-CP", "28/2020/NĐ-CP"})

    def test_retrieval_import_is_not_order_dependent(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "from legal_rag.retrieval import BM25Index, retrieve_articles"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_planner_rejects_hallucinated_doc_id_outside_corpus(self) -> None:
        data = self._plan_json("Doanh nghiệp được hỗ trợ khi nào?")
        data["target_doc_ids"] = ["99/2099/QH99"]

        plan = legal_query_plan_from_json(data, "Doanh nghiệp được hỗ trợ khi nào?")

        self.assertEqual(plan.target_doc_ids, [])
        self.assertNotIn("99/2099/QH99", plan.governing_doc_hints)
        self.assertIn("planner_doc_ids_rejected:99/2099/QH99", plan.reconciliation_issues)

    def test_planner_accepts_corpus_grounded_doc_hint_as_advisory(self) -> None:
        data = self._plan_json("Chủ doanh nghiệp tư nhân cho thuê toàn bộ doanh nghiệp phải làm gì?")
        data["target_doc_ids"] = ["59/2020/QH14"]
        candidate = CorpusCandidate(
            doc_id="59/2020/QH14",
            title="Luật 59/2020/QH14 Doanh nghiệp",
            article_labels=["Điều 191"],
            score=1.0,
        )

        plan = legal_query_plan_from_json(
            data,
            "Chủ doanh nghiệp tư nhân cho thuê toàn bộ doanh nghiệp phải làm gì?",
            corpus_candidates=[candidate],
        )

        self.assertEqual(plan.target_doc_ids, [])
        self.assertIn("59/2020/QH14", plan.governing_doc_hints)
        self.assertIn("Luật 59/2020/QH14 Doanh nghiệp", plan.governing_doc_hints)
        self.assertIn("planner_doc_ids_catalog_grounded:59/2020/QH14", plan.reconciliation_issues)

    def test_planner_accepts_article_label_only_when_present_in_question(self) -> None:
        data = self._plan_json("Theo Điều 4, doanh nghiệp được hỗ trợ khi nào?")
        data["target_article_labels"] = ["Điều 4"]

        plan = legal_query_plan_from_json(data, "Theo Điều 4, doanh nghiệp được hỗ trợ khi nào?")

        self.assertEqual(plan.target_article_labels, ["Điều 4"])

    def test_tax_land_plan_sets_guidance_flag(self) -> None:
        data = self._plan_json("Cơ sở ươm tạo được hỗ trợ gì về thuế và đất đai?")
        data["intent"] = "tax_land"
        data["needs_guidance_docs"] = False

        plan = legal_query_plan_from_json(data, "Cơ sở ươm tạo được hỗ trợ gì về thuế và đất đai?")

        self.assertTrue(plan.needs_guidance_docs)
        self.assertEqual(plan.intent, "tax_land")
        self.assertIn("thuế", plan.filters["must_include_terms"])
        self.assertIn("đất đai", plan.filters["must_include_terms"])
        self.assertIn("support_policy", plan.target_norm_roles)

    def test_labor_question_does_not_hardcode_domain_synonyms(self) -> None:
        data = self._plan_json("Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào?")
        data["intent"] = "penalty"
        data["needs_guidance_docs"] = False

        plan = legal_query_plan_from_json(
            data,
            "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào?",
        )

        self.assertNotIn("văn bằng", plan.filters["must_include_terms"])
        self.assertNotIn("chứng chỉ", plan.filters["must_include_terms"])
        self.assertIn("mức phạt", plan.requested_components)

    def test_domain_anchor_names_are_not_promoted_to_document_hints(self) -> None:
        hints = infer_governing_doc_hints(
            "Cơ sở ươm tạo được hỗ trợ gì về thuế và đất đai?",
            domain_anchors=["support_policy_sme", "tax_admin_penalties"],
        )

        self.assertEqual(hints, ["support_policy_sme", "tax_admin_penalties"])

    def test_tax_invoice_authority_question_does_not_require_penalty_amount(self) -> None:
        metadata = infer_runtime_metadata(
            "Cơ quan nào có thẩm quyền xử phạt vi phạm hành chính về thuế và hóa đơn?",
            intent="authority",
        )

        self.assertEqual(metadata.question_type, "authority")
        self.assertIn("authority", metadata.target_norm_roles)
        self.assertNotIn("penalty", metadata.target_norm_roles)
        self.assertIn("hóa đơn", metadata.requested_components)
        self.assertNotIn("mức phạt", metadata.requested_components)

    def test_local_documents_are_downranked_for_national_question(self) -> None:
        national = self._article("Điều 4", doc_id="80/2021/NĐ-CP")
        national.doc_type = "Nghị định"
        national.title_for_submission = "Nghị định số 80/2021/NĐ-CP Quy định chi tiết Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
        local = self._article("Điều 4", doc_id="09/2020/NQ-HĐND")
        local.doc_type = "Nghị quyết"
        local.title_for_submission = "Nghị quyết số 09/2020/NQ-HĐND Quy định chính sách hỗ trợ doanh nghiệp nhỏ và vừa trên địa bàn tỉnh Yên Bái"
        national.score = local.score = 1.0
        plan = LegalQueryPlan(
            intent="support_policy",
            question_scope="unknown",
            normalized_question="cơ sở ươm tạo được hỗ trợ thuế đất đai",
            question_type="list",
            legal_terms=["cơ sở ươm tạo", "thuế", "đất đai"],
            legal_facets=["hỗ trợ", "doanh nghiệp nhỏ và vừa"],
            domain_anchors=["support_policy_sme"],
            queries=[PlannedQuery("core", "cơ sở ươm tạo hỗ trợ thuế đất đai")],
        )

        ranked = _apply_retrieval_bias([local, national], plan)
        guarded, removed = _suppress_unrequested_local_scope(
            ranked,
            plan,
            question="Cơ sở ươm tạo được hỗ trợ thuế đất đai?",
        )

        self.assertEqual(ranked[0].doc_id, "80/2021/NĐ-CP")
        self.assertEqual([article.doc_id for article in guarded], ["80/2021/NĐ-CP"])
        self.assertEqual(removed, {"09/2020/NQ-HĐND"})

    def test_structured_bias_cannot_overturn_large_reranker_margin(self) -> None:
        strong = self._article("Điều 191", doc_id="59/2020/QH14")
        strong.metadata.update({"rerank_score": 0.9, "fusion_score": 0.02, "norm_roles": []})
        strong.score = 0.9
        weak = self._article("Điều 4", doc_id="01/2020/QH14")
        weak.article_title = "Thủ tục và thẩm quyền"
        weak.text = "Thủ tục, hồ sơ, thẩm quyền, trách nhiệm và điều kiện hỗ trợ."
        weak.metadata.update({"rerank_score": 0.2, "fusion_score": 0.04, "norm_roles": ["procedure", "authority"]})
        weak.score = 0.2
        plan = LegalQueryPlan(
            intent="procedure",
            question_scope="unknown",
            normalized_question="cho thuê toàn bộ doanh nghiệp tư nhân phải thông báo thế nào",
            requested_components=["trình tự"],
            retrieval_bias="procedure_articles",
            filters={"must_include_terms": ["thủ tục"], "should_include_terms": ["thẩm quyền"]},
            queries=[PlannedQuery("original", "cho thuê toàn bộ doanh nghiệp tư nhân")],
        )

        ranked = _apply_retrieval_bias([strong, weak], plan, HybridRetrievalConfig())

        self.assertEqual(ranked[0].article_key, strong.article_key)

    def test_tax_authority_cross_reference_is_downranked_below_direct_regime_doc(self) -> None:
        pointer = self._article("Điều 20", doc_id="236/2025/NĐ-CP")
        pointer.doc_type = "Nghị định"
        pointer.title_for_submission = (
            "Nghị định số 236/2025/NĐ-CP Quy định chi tiết áp dụng thuế thu nhập doanh nghiệp bổ sung "
            "theo quy định chống xói mòn cơ sở thuế toàn cầu"
        )
        pointer.text = (
            "Điều 20. Xử phạt hành vi vi phạm hành chính về thuế\n"
            "Các nội dung về thẩm quyền xử phạt được thực hiện theo quy định của pháp luật về xử phạt vi phạm hành chính về thuế, hóa đơn."
        )
        direct = self._article("Điều 15", doc_id="125/2020/NĐ-CP")
        direct.doc_type = "Nghị định"
        direct.title_for_submission = "Nghị định số 125/2020/NĐ-CP Quy định xử phạt vi phạm hành chính về thuế, hóa đơn"
        direct.text = "Điều 15. Thẩm quyền xử phạt\nCục trưởng Cục Thuế có thẩm quyền xử phạt vi phạm hành chính về thuế, hóa đơn."
        pointer.score = direct.score = 1.0
        plan = LegalQueryPlan(
            intent="authority",
            question_scope="unknown",
            normalized_question="cơ quan nào có thẩm quyền xử phạt vi phạm hành chính về thuế và hóa đơn",
            question_type="authority",
            answer_shape="document_pointer",
            legal_facets=["thuế", "hóa đơn", "xử phạt", "thẩm quyền"],
            requested_components=["thẩm quyền", "cơ quan", "thuế", "hóa đơn"],
            target_norm_roles=["authority"],
            governing_doc_hints=["xử phạt vi phạm hành chính thuế", "xử phạt vi phạm hành chính hóa đơn"],
            domain_anchors=["tax_admin_penalties"],
            retrieval_bias="authority_articles",
            queries=[PlannedQuery("core", "thẩm quyền xử phạt vi phạm hành chính thuế hóa đơn")],
        )

        ranked = _apply_retrieval_bias([pointer, direct], plan)

        self.assertEqual(ranked[0].doc_id, "125/2020/NĐ-CP")

    def test_hybrid_search_with_plan_queries_dense_and_sparse_for_semantic_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            articles, index_path, _articles_path = self._build_inputs(root)
            client = CountingQdrantClient([{"payload": article.to_dict(), "score": 0.5} for article in articles])
            retriever = HybridRetriever(
                articles,
                BM25Index.load(index_path),
                HybridRetrievalConfig(rerank_top_k=5, final_top_k=2),
                qdrant_client=client,
                embedder=FakeEmbedder(),
                reranker=FakeReranker(),
            )
            plan = LegalQueryPlan(
                intent="condition",
                question_scope="single_article",
                normalized_question="điều kiện hỗ trợ doanh nghiệp",
                queries=[
                    PlannedQuery("original", "Doanh nghiệp được hỗ trợ khi nào?"),
                    PlannedQuery("legal_terms", "điều kiện hỗ trợ doanh nghiệp"),
                ],
            )

            hits = retriever.search_with_plan("Doanh nghiệp được hỗ trợ khi nào?", plan, top_k=2)

            self.assertTrue(hits)
            self.assertEqual(client.queries.count("dense"), 1)
            self.assertEqual(client.queries.count("sparse"), 1)

    def test_planner_repairs_to_compact_role_queries(self) -> None:
        question = "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?"
        data = self._plan_json(question)
        data["intent"] = "tax_land"
        data["legal_terms"] = ["cơ sở ươm tạo", "khu làm việc chung", "hỗ trợ", "thuế", "đất đai"]
        data["requested_components"] = ["cơ quan"]
        data["queries"] = [{"kind": "original", "text": question, "purpose": "preserve user wording"}]

        plan = legal_query_plan_from_json(data, question)

        self.assertNotIn("cơ quan", plan.requested_components)
        self.assertIn("thuế", plan.requested_components)
        self.assertIn("đất đai", plan.requested_components)
        self.assertLessEqual(len(plan.queries), 4)
        self.assertTrue(any(query.kind == "core" for query in plan.queries))
        self.assertTrue(any(query.kind in {"component", "guidance"} for query in plan.queries))
        self.assertTrue(all(len(query.text.split()) <= 22 for query in plan.queries))
        self.assertFalse(any(query.text.startswith(question + " ") for query in plan.queries))

    def test_legal_lexicon_expands_lay_sme_surface_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            article = self._article("Điều 22", doc_id="80/2021/NĐ-CP")
            article.title_for_submission = "Nghị định số 80/2021/NĐ-CP Quy định chi tiết Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
            article.article_title = "Nội dung hỗ trợ doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo"
            article.text = (
                "Điều 22. Nội dung hỗ trợ doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo\n"
                "b) Hỗ trợ tối đa 50% chi phí thuê mặt bằng tại các cơ sở ươm tạo, khu làm việc chung "
                "nhưng không quá 5 triệu đồng/tháng/doanh nghiệp. Thời gian hỗ trợ tối đa là 03 năm "
                "kể từ ngày doanh nghiệp ký hợp đồng thuê mặt bằng."
            )
            articles_path = root / "articles.jsonl"
            write_articles_jsonl([article], articles_path)
            lexicon_path = root / "legal_lexicon.jsonl"

            build_legal_lexicon(articles_path, lexicon_path)
            matches = search_legal_lexicon(
                load_legal_lexicon(lexicon_path),
                "Công ty nhỏ và vừa được hỗ trợ giá thuê mặt bằng sản xuất trong thời gian tối đa là bao lâu?",
            )
            terms = " ".join(match.term for match in matches)

            self.assertIn("doanh nghiệp nhỏ và vừa", terms)
            self.assertIn("thuê mặt bằng", terms)
            self.assertTrue(any("thời gian hỗ trợ" in match.term for match in matches))

    def test_planner_uses_lexicon_terms_for_sme_rental_duration_query(self) -> None:
        question = "Công ty nhỏ và vừa được hỗ trợ giá thuê mặt bằng sản xuất trong thời gian tối đa là bao lâu?"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            article = self._article("Điều 22", doc_id="80/2021/NĐ-CP")
            article.title_for_submission = "Nghị định số 80/2021/NĐ-CP Quy định chi tiết Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
            article.article_title = "Nội dung hỗ trợ doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo"
            article.text = (
                "Hỗ trợ tối đa 50% chi phí thuê mặt bằng tại các cơ sở ươm tạo, khu làm việc chung. "
                "Thời gian hỗ trợ tối đa là 03 năm kể từ ngày doanh nghiệp ký hợp đồng thuê mặt bằng."
            )
            articles_path = root / "articles.jsonl"
            write_articles_jsonl([article], articles_path)
            lexicon_path = root / "legal_lexicon.jsonl"
            build_legal_lexicon(articles_path, lexicon_path)
            candidates = search_legal_lexicon(load_legal_lexicon(lexicon_path), question)
            data = self._plan_json(question)
            data["intent"] = "support_policy"
            data["queries"] = [{"kind": "core", "text": "hỗ trợ", "purpose": "too broad"}]

            plan = legal_query_plan_from_json(data, question, lexicon_candidates=candidates)
            query_text = " ".join(query.text for query in plan.queries).lower()

            self.assertIn("doanh nghiệp nhỏ và vừa", query_text)
            self.assertIn("thuê mặt bằng", query_text)
            self.assertIn("thời gian", query_text)
            self.assertNotIn("80/2021/NĐ-CP", plan.governing_doc_hints)
            self.assertTrue(any("Luật Hỗ trợ doanh nghiệp nhỏ và vừa" in hint for hint in plan.governing_doc_hints))

    def test_micro_chunk_build_promotes_to_parent_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_text = "Điều 4. Điều kiện hỗ trợ\n" + "Nội dung hỗ trợ doanh nghiệp. " * 220
            article = ArticleNode(
                article_key="01/2020/QH14|Luật 01/2020/QH14 Luật X|Điều 4",
                doc_id="01/2020/QH14",
                doc_type="Luật",
                title_for_submission="Luật 01/2020/QH14 Luật X",
                article_label="Điều 4",
                article_title="Điều kiện hỗ trợ",
                text=long_text,
                metadata={"chunk_type": "article"},
            )
            articles_path = root / "articles.jsonl"
            write_articles_jsonl([article], articles_path)
            client = CountingQdrantClient()

            report = build_hybrid_index(
                articles_path,
                HybridRetrievalConfig(
                    collection="test",
                    embedding_cache_dir=str(root / "cache"),
                    enable_micro_chunks=True,
                ),
                qdrant_client=client,
                embedder=FakeEmbedder(),
            )

            self.assertGreater(report.points, report.articles)
            payloads = [point["payload"] if isinstance(point, dict) else point.payload for point in client.upserted]
            self.assertTrue(any(payload["metadata"].get("parent_article_key") == article.article_key for payload in payloads))

    def test_answer_formatter_uses_only_selected_evidence(self) -> None:
        article4 = self._article("Điều 4")
        article5 = self._article("Điều 5")
        blocks = build_evidence_blocks([article4, article5])
        answer = evidence_answer_from_json(
            {
                "answer": "Doanh nghiệp được hỗ trợ theo Điều 4.",
                "used_evidence_ids": ["E1"],
                "insufficient_evidence": False,
                "support_map": [],
            },
            blocks,
        )

        used = articles_from_used_evidence(blocks, answer.used_evidence_ids)

        self.assertEqual([article.relevant_article for article in used], [article4.relevant_article])

    def test_answer_parser_minimizes_used_evidence_from_support_map(self) -> None:
        article4 = self._article("Điều 4")
        article5 = self._article("Điều 5")
        blocks = build_evidence_blocks([article4, article5])

        answer = evidence_answer_from_json(
            {
                "answer": "Doanh nghiệp được hỗ trợ theo Điều 4.",
                "used_evidence_ids": ["E1", "E2"],
                "insufficient_evidence": False,
                "support_map": [{"claim": "được hỗ trợ", "evidence_ids": ["E1"], "article_refs": [article4.relevant_article]}],
            },
            blocks,
        )

        self.assertEqual(answer.used_evidence_ids, ["E1"])

    def test_answer_parser_rejects_claim_backed_only_by_adjacent_evidence(self) -> None:
        article = self._article("Điều 30", doc_id="23/2023/TT-BKHCN")
        article.metadata["evidence_admission"] = {"support_type": "adjacent"}
        blocks = build_evidence_blocks([article])

        with self.assertRaisesRegex(OllamaError, "answer_claim_uses_non_direct_evidence"):
            evidence_answer_from_json(
                {
                    "answer_text": "Đơn phải có tài liệu A.",
                    "used_evidence_ids": ["E1"],
                    "insufficient_evidence": False,
                    "claims": [{"claim": "Đơn phải có tài liệu A.", "evidence_ids": ["E1"]}],
                },
                blocks,
            )

    def test_evidence_admission_marks_examination_article_adjacent_to_filing_question(self) -> None:
        examination = self._article("Điều 30", doc_id="23/2023/TT-BKHCN")
        examination.article_title = "Thẩm định nội dung đơn đăng ký"
        examination.text = "Nguồn thông tin tối thiểu để thẩm định bao gồm nhãn hiệu đã được bảo hộ."
        requirement = self._article("Điều 28", doc_id="23/2023/TT-BKHCN")
        requirement.article_title = "Yêu cầu đối với đơn đăng ký"
        requirement.text = "Đơn đăng ký phải đáp ứng các yêu cầu theo quy định."
        plan = LegalQueryPlan(
            intent="procedure",
            question_scope="unknown",
            normalized_question="nộp đơn đăng ký cần chuẩn bị tài liệu và thông tin gì",
            requested_components=["hồ sơ"],
            queries=[PlannedQuery("original", "nộp đơn đăng ký tài liệu")],
        )

        selected = _select_evidence_candidates(plan, [examination, requirement], 2)
        support_types = {item.article_label: item.metadata["evidence_admission"]["support_type"] for item in selected}

        self.assertNotIn("Điều 30", support_types)
        self.assertEqual(support_types["Điều 28"], "direct")

    def test_evidence_selection_excludes_non_direct_context(self) -> None:
        adjacent = self._article("Điều 30", doc_id="23/2023/TT-BKHCN")
        adjacent.article_title = "Thẩm định nội dung đơn đăng ký"
        adjacent.text = "Nguồn thông tin dùng để thẩm định đơn đăng ký chỉ dẫn địa lý."
        plan = LegalQueryPlan(
            intent="procedure",
            question_scope="unknown",
            normalized_question="nộp đơn đăng ký chỉ dẫn địa lý cần tài liệu gì",
            requested_components=["hồ sơ"],
            queries=[PlannedQuery("original", "đơn đăng ký chỉ dẫn địa lý tài liệu")],
        )

        selected = _select_evidence_candidates(plan, [adjacent], 2)
        answer = _insufficient_answer_for_corpus_gap(plan, build_evidence_blocks(selected))

        self.assertEqual(selected, [])
        self.assertIsNotNone(answer)
        self.assertTrue(answer.insufficient_evidence)

    def test_evidence_admission_requires_question_grounded_legal_focus(self) -> None:
        wrong_regime = self._article("Điều 76", doc_id="168/2025/NĐ-CP")
        wrong_regime.title_for_submission = "Nghị định về đăng ký doanh nghiệp"
        wrong_regime.article_title = "Hồ sơ đăng ký doanh nghiệp"
        wrong_regime.text = "Hồ sơ đăng ký doanh nghiệp phải có thông tin chính xác và đầy đủ."
        direct = self._article("Điều 28", doc_id="23/2023/TT-BKHCN")
        direct.title_for_submission = "Thông tư về thủ tục xác lập quyền sở hữu công nghiệp"
        direct.article_title = "Yêu cầu đối với đơn đăng ký chỉ dẫn địa lý"
        direct.text = "Đơn đăng ký chỉ dẫn địa lý phải có tài liệu xác định khu vực địa lý."
        plan = LegalQueryPlan(
            intent="procedure",
            question_scope="unknown",
            normalized_question="nộp đơn đăng ký chỉ dẫn địa lý cần chuẩn bị tài liệu gì",
            legal_facets=["đăng ký chỉ dẫn địa lý"],
            requested_components=["hồ sơ"],
            queries=[PlannedQuery("original", "đơn đăng ký chỉ dẫn địa lý tài liệu")],
        )

        selected = _select_evidence_candidates(plan, [wrong_regime, direct], 2)

        self.assertEqual([item.doc_id for item in selected], ["23/2023/TT-BKHCN"])
        self.assertEqual(selected[0].metadata["evidence_admission"]["support_type"], "direct")

    def test_evidence_admission_requires_requested_legal_action(self) -> None:
        wrong_action = self._article("Điều 14", doc_id="116/2009/NĐ-CP")
        wrong_action.article_title = "Quản lý, cấp phát văn bằng, chứng chỉ nghề"
        wrong_action.text = "Phạt hành vi cấp phát sai và buộc thu hồi phôi văn bằng, chứng chỉ đã in."
        correct = self._article("Điều 9", doc_id="12/2022/NĐ-CP")
        correct.article_title = "Vi phạm quy định về giao kết hợp đồng lao động"
        correct.text = (
            "Phạt tiền người sử dụng lao động giữ bản chính văn bằng hoặc chứng chỉ của người lao động. "
            "Buộc trả lại bản chính văn bằng, chứng chỉ đã giữ."
        )
        plan = LegalQueryPlan(
            intent="penalty",
            question_scope="unknown",
            normalized_question="công ty giữ bản chính bằng cấp của nhân viên thì bị phạt và khắc phục ra sao",
            legal_facets=["bản chính bằng cấp"],
            requested_components=["mức phạt", "biện pháp khắc phục"],
            entities={"subjects": ["công ty"], "actions": ["giữ"], "objects": ["bản chính bằng cấp"]},
            queries=[PlannedQuery("original", "giữ bản chính bằng cấp nhân viên")],
        )

        selected = _select_evidence_candidates(plan, [wrong_action, correct], 2)

        self.assertEqual([item.doc_id for item in selected], ["12/2022/NĐ-CP"])

    def test_corpus_gap_guard_avoids_claims_from_only_adjacent_evidence(self) -> None:
        article = self._article("Điều 34", doc_id="65/2023/NĐ-CP")
        article.metadata["support_snippet"] = "Phạm vi quyền đối với tên thương mại gồm lĩnh vực và lãnh thổ kinh doanh."
        article.metadata["evidence_admission"] = {"support_type": "adjacent"}
        plan = LegalQueryPlan(
            intent="condition",
            question_scope="unknown",
            normalized_question="điều kiện để tên thương mại được bảo hộ",
            requested_components=["điều kiện"],
            queries=[PlannedQuery("original", "điều kiện bảo hộ tên thương mại")],
        )

        answer = _insufficient_answer_for_corpus_gap(plan, build_evidence_blocks([article]))

        self.assertIsNotNone(answer)
        self.assertTrue(answer.insufficient_evidence)
        self.assertEqual(answer.used_evidence_ids, [])

    def test_answer_prompt_prefers_single_evidence_text_block(self) -> None:
        article = self._article("Điều 4")
        article.metadata["support_snippet"] = "Khoản 1. Hồ sơ gồm đơn đề nghị."
        article.metadata["support_span_label"] = "Khoản 1"
        blocks = build_evidence_blocks([article])
        plan = LegalQueryPlan(
            intent="procedure",
            question_scope="single_article",
            normalized_question="Hồ sơ gồm những gì?",
            question_type="procedure",
            answer_shape="procedure_steps",
            requested_components=["hồ sơ"],
            target_norm_roles=["procedure"],
            governing_doc_hints=["bộ luật lao động"],
            domain_anchors=["labor_sanctions"],
            queries=[PlannedQuery("original", "Hồ sơ gồm những gì?")],
        )

        prompt = _answer_user_prompt("Hồ sơ gồm những gì?", plan, blocks)

        self.assertIn("support_span: Khoản 1", prompt)
        self.assertIn("evidence_text:", prompt)
        self.assertNotIn("article_text_excerpt", prompt)
        self.assertIn("canonical_article", prompt)
        self.assertIn("không tự ghi citation/Điều luật", prompt)

    def test_support_span_prefers_proposal_dossier_over_payment_dossier(self) -> None:
        article = self._article("Điều 32", doc_id="80/2021/NĐ-CP")
        article.title_for_submission = "Nghị định số 80/2021/NĐ-CP Quy định chi tiết Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
        article.article_title = "Quy trình, thủ tục hỗ trợ"
        article.text = "Điều 32. Quy trình, thủ tục hỗ trợ"
        article.metadata["clause_nodes"] = [
            {
                "span_key": "Điều 32|Khoản 4",
                "label": "Khoản 4",
                "text": (
                    "4. Hồ sơ đề xuất nhu cầu hỗ trợ bao gồm:\n"
                    "a) Tờ khai xác định doanh nghiệp siêu nhỏ, doanh nghiệp nhỏ, doanh nghiệp vừa và đề xuất nhu cầu hỗ trợ;\n"
                    "b) Những tài liệu, hồ sơ liên quan đến nội dung đề xuất hỗ trợ (nếu có)."
                ),
                "point_nodes": [],
            },
            {
                "span_key": "Điều 32|Khoản 5",
                "label": "Khoản 5",
                "text": "5. Hồ sơ thanh toán kinh phí ngân sách nhà nước hỗ trợ gồm các hóa đơn, chứng từ tài chính có liên quan.",
                "point_nodes": [],
            },
        ]
        plan = LegalQueryPlan(
            intent="procedure",
            question_scope="unknown",
            normalized_question="hồ sơ đề nghị hỗ trợ doanh nghiệp nhỏ và vừa tham gia cụm liên kết ngành gồm những gì",
            question_type="procedure",
            answer_shape="procedure_steps",
            legal_terms=["hồ sơ", "đề nghị hỗ trợ", "cụm liên kết ngành"],
            legal_facets=["doanh nghiệp nhỏ và vừa", "hỗ trợ"],
            requested_components=["hồ sơ"],
            target_norm_roles=["procedure"],
            domain_anchors=["support_policy_sme"],
            queries=[PlannedQuery("core", "hồ sơ đề nghị hỗ trợ doanh nghiệp nhỏ và vừa cụm liên kết ngành")],
        )

        ranked = _apply_retrieval_bias([article], plan)
        snippet = ranked[0].metadata.get("support_snippet", "")

        self.assertIn("Tờ khai xác định doanh nghiệp", snippet)
        self.assertIn("tài liệu, hồ sơ liên quan", snippet)
        self.assertNotIn("hóa đơn, chứng từ", snippet)

    def test_support_span_overrides_existing_point_snippet_for_list_question(self) -> None:
        article = self._article("Điều 32", doc_id="80/2021/NĐ-CP")
        article.title_for_submission = "Nghị định số 80/2021/NĐ-CP Quy định chi tiết Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
        article.article_title = "Quy trình, thủ tục hỗ trợ"
        article.text = (
            "Điều 32. Quy trình, thủ tục hỗ trợ\n"
            "4. Hồ sơ đề xuất nhu cầu hỗ trợ bao gồm:\n"
            "a) Tờ khai xác định doanh nghiệp siêu nhỏ, doanh nghiệp nhỏ, doanh nghiệp vừa và đề xuất nhu cầu hỗ trợ;\n"
            "b) Những tài liệu, hồ sơ liên quan đến nội dung đề xuất hỗ trợ (nếu có).\n"
            "5. Hồ sơ thanh toán kinh phí gồm hóa đơn, chứng từ tài chính."
        )
        article.metadata["support_snippet"] = "a) Tờ khai xác định doanh nghiệp siêu nhỏ, doanh nghiệp nhỏ, doanh nghiệp vừa."
        article.metadata["support_span_key"] = "Điều 32|Khoản 4|Điểm a"
        article.metadata["support_span_label"] = "Điểm a"
        plan = LegalQueryPlan(
            intent="procedure",
            question_scope="unknown",
            normalized_question="hồ sơ đề nghị hỗ trợ doanh nghiệp nhỏ và vừa gồm những gì",
            question_type="procedure",
            answer_shape="procedure_steps",
            legal_terms=["hồ sơ", "đề nghị hỗ trợ"],
            legal_facets=["doanh nghiệp nhỏ và vừa", "hỗ trợ"],
            requested_components=["hồ sơ"],
            target_norm_roles=["procedure"],
            domain_anchors=["support_policy_sme"],
            queries=[PlannedQuery("core", "hồ sơ đề nghị hỗ trợ doanh nghiệp nhỏ và vừa")],
        )

        ranked = _apply_retrieval_bias([article], plan)
        snippet = ranked[0].metadata.get("support_snippet", "")

        self.assertIn("Tờ khai xác định doanh nghiệp", snippet)
        self.assertIn("tài liệu, hồ sơ liên quan", snippet)
        self.assertEqual(ranked[0].metadata.get("support_span_label"), "Khoản 4")

    def test_support_span_recomputes_incomplete_penalty_remedy_snippet(self) -> None:
        article = self._article("Điều 9", doc_id="12/2022/NĐ-CP")
        article.title_for_submission = "Nghị định số 12/2022/NĐ-CP Quy định xử phạt vi phạm hành chính trong lĩnh vực lao động"
        article.article_title = "Vi phạm quy định về giao kết hợp đồng lao động"
        article.metadata["support_snippet"] = (
            "3. Biện pháp khắc phục hậu quả\n"
            "d) Buộc người sử dụng lao động trả lại bản chính giấy tờ tùy thân; văn bằng; chứng chỉ đã giữ."
        )
        article.metadata["support_span_key"] = "Điều 9|Khoản 3"
        article.metadata["support_span_label"] = "Khoản 3"
        article.metadata["clause_nodes"] = [
            {
                "span_key": "Điều 9|Khoản 2",
                "label": "Khoản 2",
                "text": (
                    "2. Phạt tiền từ 20.000.000 đồng đến 25.000.000 đồng đối với người sử dụng lao động "
                    "có một trong các hành vi sau đây:\n"
                    "a) Giữ bản chính giấy tờ tùy thân, văn bằng hoặc chứng chỉ của người lao động khi giao kết "
                    "hoặc thực hiện hợp đồng lao động;"
                ),
                "point_nodes": [
                    {
                        "span_key": "Điều 9|Khoản 2|Điểm a",
                        "label": "Điểm a",
                        "text": (
                            "a) Giữ bản chính giấy tờ tùy thân, văn bằng hoặc chứng chỉ của người lao động khi "
                            "giao kết hoặc thực hiện hợp đồng lao động;"
                        ),
                    },
                    {
                        "span_key": "Điều 9|Khoản 2|Điểm b",
                        "label": "Điểm b",
                        "text": "b) Buộc người lao động thực hiện biện pháp bảo đảm bằng tiền hoặc tài sản khác.",
                    },
                ],
            },
            {
                "span_key": "Điều 9|Khoản 3",
                "label": "Khoản 3",
                "text": (
                    "3. Biện pháp khắc phục hậu quả\n"
                    "d) Buộc người sử dụng lao động trả lại bản chính giấy tờ tùy thân; văn bằng; chứng chỉ đã giữ "
                    "của người lao động đối với hành vi vi phạm quy định tại điểm a khoản 2 Điều này;"
                ),
                "point_nodes": [
                    {
                        "span_key": "Điều 9|Khoản 3|Điểm d",
                        "label": "Điểm d",
                        "text": (
                            "d) Buộc người sử dụng lao động trả lại bản chính giấy tờ tùy thân; văn bằng; chứng chỉ "
                            "đã giữ của người lao động đối với hành vi vi phạm quy định tại điểm a khoản 2 Điều này;"
                        ),
                    },
                    {
                        "span_key": "Điều 9|Khoản 3|Điểm đ",
                        "label": "Điểm đ",
                        "text": "đ) Buộc trả lại số tiền hoặc tài sản đã giữ của người lao động cộng với tiền lãi.",
                    },
                ],
            },
        ]
        plan = LegalQueryPlan(
            intent="penalty",
            question_scope="unknown",
            normalized_question="công ty giữ bản chính bằng cấp nhân viên khi ký hợp đồng bị xử lý và khắc phục ra sao",
            question_type="penalty",
            answer_shape="single_rule",
            legal_terms=["giữ bản chính", "bằng cấp"],
            legal_facets=["giữ bản chính", "văn bằng", "chứng chỉ"],
            requested_components=["mức phạt", "biện pháp khắc phục"],
            target_norm_roles=["penalty"],
            domain_anchors=["labor_sanctions"],
            retrieval_bias="sanction_articles",
            queries=[PlannedQuery("core", "giữ bản chính bằng cấp 12/2022/NĐ-CP")],
        )

        enriched = _attach_plan_support_span(article, plan)
        snippet = enriched.metadata.get("support_snippet", "")

        self.assertIn("Phạt tiền từ 20.000.000 đồng đến 25.000.000 đồng", snippet)
        self.assertIn("trả lại bản chính giấy tờ tùy thân", snippet)
        self.assertNotIn("trả lại số tiền hoặc tài sản", snippet)
        self.assertIn("Khoản 2", enriched.metadata.get("support_span_label", ""))
        self.assertIn("Khoản 3", enriched.metadata.get("support_span_label", ""))

    def test_dedupe_prefers_richer_parent_article_over_point_hit(self) -> None:
        parent = self._article("Điều 32", doc_id="80/2021/NĐ-CP")
        parent.text = "Điều 32. Quy trình\n4. Hồ sơ đề xuất nhu cầu hỗ trợ bao gồm:\na) Tờ khai;\nb) Tài liệu."
        parent.metadata["clause_nodes"] = [{"span_key": "Điều 32|Khoản 4", "label": "Khoản 4", "text": "4. Hồ sơ gồm:\na) Tờ khai;\nb) Tài liệu.", "point_nodes": []}]
        point = self._article("Điều 32", doc_id="80/2021/NĐ-CP")
        point.article_key = f"{parent.article_key}::point:Điều 32|Khoản 4|Điểm a"
        point.text = "a) Tờ khai;"
        point.score = 2.0
        point.metadata = {
            "node_type": "point",
            "parent_article_key": parent.article_key,
            "parent_article_title": parent.article_title,
            "support_snippet": point.text,
            "support_span_key": "Điều 32|Khoản 4|Điểm a",
            "support_span_label": "Điểm a",
        }
        parent.score = 1.0

        merged = _promote_and_dedupe([point, parent])

        self.assertEqual(len(merged), 1)
        self.assertIn("Tài liệu", merged[0].text)
        self.assertEqual(merged[0].score, 2.0)

    def test_verifier_detects_ambiguous_same_article_label(self) -> None:
        a = self._article("Điều 5", doc_id="01/2020/QH14")
        b = self._article("Điều 5", doc_id="02/2020/QH14")

        result = verify_used_evidence_answer("Căn cứ Điều 5, doanh nghiệp phải nộp hồ sơ.", [a, b], [a, b])

        self.assertFalse(result.ok)
        self.assertIn("ambiguous_article_label:điều 5", result.issues)

    def test_verifier_detects_semantic_domain_mismatch(self) -> None:
        article = self._article("Điều 4", doc_id="142/2025/QH15")
        article.title_for_submission = "Luật 142/2025/QH15 PHỤC HỒI, PHÁ SẢN"
        article.text = "Điều 4. Thủ tục phục hồi, phá sản doanh nghiệp."

        result = verify_used_evidence_answer(
            "Theo Điều 4 Luật 142/2025/QH15 PHỤC HỒI, PHÁ SẢN.",
            [article],
            [article],
            question="Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?",
        )

        self.assertTrue(result.ok)
        self.assertTrue(any(issue.startswith("semantic_domain_mismatch") for issue in result.warnings))

    def test_verifier_detects_missing_required_component(self) -> None:
        article = self._article("Điều 5")
        article.text = "Điều 5. Hồ sơ\nHồ sơ gồm đơn đề nghị và tài liệu liên quan. Thời hạn giải quyết là 15 ngày."
        article.metadata["norm_roles"] = ["procedure"]

        result = verify_used_evidence_answer(
            "Theo Điều 5, doanh nghiệp nộp hồ sơ.",
            [article],
            [article],
            question="Hồ sơ đề nghị gồm những gì và thời hạn giải quyết là bao lâu?",
            required_components=["hồ sơ", "thời hạn"],
            target_norm_roles=["procedure"],
            covered_components=["hồ sơ"],
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.needs_repair)
        self.assertIn("component_coverage_missing:thời hạn", result.issues)

    def test_verifier_rejects_specific_claim_value_missing_from_evidence(self) -> None:
        article = self._article("Điều 22")
        article.text = "Thời gian hỗ trợ tối đa là 03 năm."
        article.metadata["support_snippet"] = article.text

        result = verify_used_evidence_answer(
            "Thời gian hỗ trợ là 05 năm. Căn cứ Điều 22.",
            [article],
            [article],
            claims=[{"claim": "Thời gian hỗ trợ là 05 năm.", "evidence_ids": ["E1"]}],
            evidence_by_id={"E1": article},
        )

        self.assertFalse(result.ok)
        self.assertIn("claim_value_not_supported:1:05 năm", result.hard_issues)

    def test_verifier_allows_explicit_partial_insufficient_component(self) -> None:
        article = self._article("Điều 22")
        article.text = "Điều 22. Hỗ trợ\nHỗ trợ tối đa 50% chi phí thuê mặt bằng cho doanh nghiệp."
        article.metadata["support_snippet"] = article.text

        result = verify_used_evidence_answer(
            "Theo Điều 22, doanh nghiệp được hỗ trợ thuê mặt bằng; về thuế, chưa tìm thấy căn cứ trong evidence được cung cấp.",
            [article],
            [article],
            question="Doanh nghiệp được hỗ trợ gì về thuế và đất đai?",
            required_components=["thuế", "đất đai"],
            insufficient_components=["thuế"],
        )

        self.assertTrue(result.ok, result.issues)

        silent = verify_used_evidence_answer(
            "Theo Điều 22, doanh nghiệp được hỗ trợ thuê mặt bằng.",
            [article],
            [article],
            question="Doanh nghiệp được hỗ trợ gì về thuế và đất đai?",
            required_components=["thuế", "đất đai"],
            insufficient_components=["thuế"],
        )
        self.assertTrue(silent.ok)
        self.assertTrue(silent.needs_repair)
        self.assertIn("component_coverage_missing:thuế", silent.issues)

    def test_verifier_does_not_trust_model_claimed_component_coverage(self) -> None:
        article = self._article("Điều 4")
        article.text = "Điều 4. Biện pháp khắc phục\nBuộc trả lại bản chính giấy tờ đã giữ."

        result = verify_used_evidence_answer(
            "Theo Điều 4, công ty phải trả lại bản chính giấy tờ đã giữ.",
            [article],
            [article],
            question="Giữ bản chính bằng cấp bị phạt bao nhiêu và khắc phục ra sao?",
            required_components=["mức phạt", "biện pháp khắc phục"],
            target_norm_roles=["penalty"],
            covered_components=["mức phạt", "biện pháp khắc phục"],
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.needs_repair)
        self.assertIn("component_coverage_missing:mức phạt", result.issues)

    def test_verifier_requires_required_component_in_answer_and_evidence(self) -> None:
        article = self._article("Điều 191", doc_id="59/2020/QH14")
        article.text = (
            "Quyền, nghĩa vụ và trách nhiệm của chủ sở hữu và người thuê đối với hoạt động kinh doanh "
            "được quy định trong hợp đồng cho thuê."
        )

        result = verify_used_evidence_answer(
            "Chủ doanh nghiệp vẫn phải chịu trách nhiệm trước pháp luật.",
            [article],
            [article],
            question="Chủ sở hữu và người thuê có những quyền và nghĩa vụ nào?",
            required_components=["quyền và nghĩa vụ"],
            covered_components=["quyền và nghĩa vụ"],
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.needs_repair)
        self.assertIn("component_coverage_missing:quyền và nghĩa vụ", result.issues)

    def test_deterministic_component_repair_appends_direct_grounded_sentence(self) -> None:
        article = self._article("Điều 191", doc_id="59/2020/QH14")
        article.text = (
            "Chủ doanh nghiệp vẫn chịu trách nhiệm trước pháp luật. "
            "Quyền, nghĩa vụ và trách nhiệm của chủ sở hữu và người thuê được quy định trong hợp đồng cho thuê."
        )
        block = build_evidence_blocks([article])[0]
        answer = type(
            "Answer",
            (),
            {
                "answer": "Chủ doanh nghiệp vẫn chịu trách nhiệm trước pháp luật.",
                "covered_components": [],
                "used_evidence_ids": [],
                "claims": [],
                "support_map": [],
            },
        )()

        repaired = _repair_missing_components_from_direct_evidence(
            answer,
            [block],
            ["quyền và nghĩa vụ"],
        )

        self.assertEqual(repaired, ["quyền và nghĩa vụ"])
        self.assertIn("Quyền, nghĩa vụ và trách nhiệm", answer.answer)
        self.assertEqual(answer.used_evidence_ids, ["E1"])
        self.assertEqual(answer.claims[0]["evidence_ids"], ["E1"])
        used = finalize_evidence_answer(answer, [block])
        result = verify_used_evidence_answer(
            answer.answer,
            used,
            [article],
            required_components=["quyền và nghĩa vụ"],
            covered_components=answer.covered_components,
            claims=answer.claims,
            evidence_by_id={"E1": article},
        )
        self.assertNotIn("component_coverage_missing:quyền và nghĩa vụ", result.issues)

    def test_verifier_rejects_authority_answer_from_cross_reference_only_evidence(self) -> None:
        article = self._article("Điều 20", doc_id="236/2025/NĐ-CP")
        article.title_for_submission = "Nghị định số 236/2025/NĐ-CP Quy định thuế thu nhập doanh nghiệp bổ sung"
        article.text = (
            "Điều 20. Xử phạt hành vi vi phạm hành chính về thuế\n"
            "Các nội dung về thẩm quyền xử phạt, mức tiền xử phạt và thủ tục xử phạt được thực hiện "
            "theo quy định của pháp luật về xử phạt vi phạm hành chính về thuế, hóa đơn."
        )
        article.metadata["support_snippet"] = article.text

        result = verify_used_evidence_answer(
            "Cơ quan có thẩm quyền xử phạt là cơ quan quản lý thuế theo Điều 20.",
            [article],
            [article],
            question="Cơ quan nào có thẩm quyền xử phạt vi phạm hành chính về thuế và hóa đơn?",
            required_components=["thẩm quyền", "cơ quan", "thuế", "hóa đơn"],
            target_norm_roles=["authority"],
            covered_components=["thẩm quyền", "cơ quan", "thuế", "hóa đơn"],
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.needs_repair)
        self.assertIn("authority_evidence_is_cross_reference_only", result.issues)

    def test_cli_repairs_dossier_list_answer_from_used_evidence(self) -> None:
        article = self._article("Điều 32", doc_id="80/2021/NĐ-CP")
        article.metadata["support_snippet"] = (
            "4. Hồ sơ đề xuất nhu cầu hỗ trợ bao gồm:\n"
            "a) Tờ khai xác định doanh nghiệp siêu nhỏ, doanh nghiệp nhỏ, doanh nghiệp vừa và đề xuất nhu cầu hỗ trợ;\n"
            "b) Những tài liệu, hồ sơ liên quan đến nội dung đề xuất hỗ trợ (nếu có)."
        )
        answer = type(
            "Answer",
            (),
            {
                "answer": "Theo Điều 32, hồ sơ gồm hồ sơ đề xuất nhu cầu hỗ trợ.",
                "covered_components": [],
            },
        )()

        changed = _repair_list_answer_from_used_evidence(
            "Hồ sơ đề nghị hỗ trợ doanh nghiệp nhỏ và vừa gồm những gì?",
            answer,
            [article],
        )

        self.assertTrue(changed)
        self.assertIn("Tờ khai xác định doanh nghiệp", answer.answer)
        self.assertIn("tài liệu, hồ sơ liên quan", answer.answer)

    def test_cli_repairs_dossier_list_from_full_article_when_snippet_is_too_narrow(self) -> None:
        article = self._article("Điều 32", doc_id="80/2021/NĐ-CP")
        article.metadata["support_snippet"] = "1. Doanh nghiệp gửi Hồ sơ đề xuất nhu cầu hỗ trợ tới cơ quan hỗ trợ."
        article.text = (
            "Điều 32. Quy trình, thủ tục hỗ trợ\n"
            "1. Doanh nghiệp gửi Hồ sơ đề xuất nhu cầu hỗ trợ tới cơ quan hỗ trợ.\n"
            "3. Đối với nội dung hỗ trợ doanh nghiệp nhỏ và vừa tham gia cụm liên kết ngành, chuỗi giá trị được thực hiện theo quy trình, thủ tục như sau:\n"
            "a) Trong thời hạn 14 ngày làm việc kể từ ngày nhận được Hồ sơ đề xuất nhu cầu hỗ trợ, cơ quan hỗ trợ xem xét hồ sơ;\n"
            "b) Trường hợp cơ quan hỗ trợ có khả năng cung cấp trực tiếp sản phẩm, dịch vụ hỗ trợ.\n"
            "4. Hồ sơ đề xuất nhu cầu hỗ trợ bao gồm:\n"
            "a) Tờ khai xác định doanh nghiệp siêu nhỏ, doanh nghiệp nhỏ, doanh nghiệp vừa và đề xuất nhu cầu hỗ trợ;\n"
            "b) Những tài liệu, hồ sơ liên quan đến nội dung đề xuất hỗ trợ (nếu có).\n"
            "5. Hồ sơ thanh toán kinh phí bao gồm:\n"
            "a) Thông báo hỗ trợ;\n"
            "b) Hợp đồng;\n"
            "c) Các hóa đơn, chứng từ tài chính có liên quan."
        )
        answer = type(
            "Answer",
            (),
            {
                "answer": "Hồ sơ gồm hồ sơ đề xuất nhu cầu hỗ trợ theo quy định.",
                "covered_components": [],
            },
        )()

        changed = _repair_list_answer_from_used_evidence(
            "Hồ sơ đề nghị hỗ trợ doanh nghiệp nhỏ và vừa tham gia cụm liên kết ngành gồm những gì?",
            answer,
            [article],
        )

        self.assertTrue(changed)
        self.assertIn("Tờ khai xác định doanh nghiệp", answer.answer)
        self.assertIn("tài liệu, hồ sơ liên quan", answer.answer)
        self.assertNotIn("hóa đơn, chứng từ", answer.answer)

    def test_cli_repairs_nested_amendment_citation_to_canonical_article(self) -> None:
        article = self._article("Điều 1", doc_id="46/2024/NĐ-CP")
        article.text = "Điều 1. Sửa đổi, bổ sung Điều 6 của Nghị định số 99/2013/NĐ-CP."
        answer = type(
            "Answer",
            (),
            {"answer": "Hành vi này bị xử lý theo Điều 6.", "covered_components": []},
        )()

        changed = _repair_citations_to_used_evidence(
            answer,
            [article],
            ["citation_not_in_used_evidence:điều 6", f"used_evidence_not_cited:{article.article_key}"],
        )

        self.assertTrue(changed)
        self.assertIn("Điều 1", answer.answer)
        self.assertNotIn("Điều 6", answer.answer)

    def test_cli_appends_missing_single_evidence_citation(self) -> None:
        article = self._article("Điều 32", doc_id="80/2021/NĐ-CP")
        answer = type(
            "Answer",
            (),
            {"answer": "Hồ sơ gồm hồ sơ đề xuất nhu cầu hỗ trợ theo quy định.", "covered_components": []},
        )()

        changed = _repair_citations_to_used_evidence(
            answer,
            [article],
            ["answer_missing_article_citation", f"used_evidence_not_cited:{article.article_key}"],
        )

        self.assertTrue(changed)
        self.assertIn("Điều 32", answer.answer)

    def test_system_builds_canonical_citation_from_claim_evidence(self) -> None:
        article = self._article("Điều 1", doc_id="46/2024/NĐ-CP")
        blocks = build_evidence_blocks([article])
        answer = evidence_answer_from_json(
            {
                "answer_text": "Hành vi này phải thực hiện theo quy định áp dụng.",
                "used_evidence_ids": ["E1"],
                "insufficient_evidence": False,
                "claims": [{"claim": "Quy định áp dụng", "evidence_ids": ["E1"], "article_refs": []}],
                "covered_components": [],
                "insufficient_components": [],
            },
            blocks,
        )

        used = finalize_evidence_answer(answer, blocks)

        self.assertEqual([item.article_key for item in used], [article.article_key])
        self.assertIn("Căn cứ Điều 1 46/2024/NĐ-CP", answer.answer)

    @staticmethod
    def _plan_json(question: str) -> dict:
        return {
            "intent": "condition",
            "question_scope": "single_article",
            "normalized_question": question,
            "legal_terms": ["doanh nghiệp", "hỗ trợ"],
            "entities": {"subjects": [], "actions": [], "conditions": [], "amounts_or_deadlines": []},
            "target_doc_ids": [],
            "target_doc_aliases": [],
            "target_article_labels": [],
            "queries": [{"kind": "original", "text": question, "purpose": "preserve user wording"}],
            "filters": {"doc_types": [], "must_include_terms": [], "should_include_terms": []},
            "needs_guidance_docs": False,
            "multi_hop_targets": [],
            "missing_facts": [],
            "confidence": 0.7,
        }

    @staticmethod
    def _article(label: str, doc_id: str = "01/2020/QH14") -> ArticleNode:
        return ArticleNode(
            article_key=f"{doc_id}|Luật {doc_id} Luật X|{label}",
            doc_id=doc_id,
            doc_type="Luật",
            title_for_submission=f"Luật {doc_id} Luật X",
            article_label=label,
            article_title="Hồ sơ",
            text=f"{label}. Hồ sơ\nDoanh nghiệp nộp hồ sơ.",
        )

    @staticmethod
    def _build_inputs(root: Path):
        raw_doc = {
            "doc_id": "01/2020/QH14",
            "doc_type": "Luật",
            "trich_yeu": "Luật X",
            "title_for_submission": "Luật 01/2020/QH14 Luật X",
            "raw_text": "Điều 4. Điều kiện hỗ trợ\nDoanh nghiệp được hỗ trợ khi đáp ứng tiêu chí.\n\nĐiều 5. Hồ sơ\nHồ sơ gồm đơn đề nghị.",
        }
        corpus_path = root / "corpus.json"
        corpus_path.write_text(json.dumps([raw_doc], ensure_ascii=False), encoding="utf-8")
        articles, warnings = ingest_corpus(corpus_path)
        if warnings:
            raise AssertionError(warnings)
        articles_path = root / "articles.jsonl"
        write_articles_jsonl(articles, articles_path)
        index_path = root / "index.json"
        BM25Index(articles).save(index_path)
        return articles, index_path, articles_path


if __name__ == "__main__":
    unittest.main()
