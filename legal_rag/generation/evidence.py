from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from legal_rag.generation.ollama import OllamaConfig, OllamaError, parse_json_response, request_ollama_chat
from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import compact_snippet


@dataclass(slots=True)
class EvidenceBlock:
    evidence_id: str
    article: ArticleNode
    support_snippet: str
    retrieval_trace: str = ""
    node_type: str = "article"
    norm_roles: list[str] = field(default_factory=list)
    support_span: str = ""

    @property
    def canonical_article(self) -> str:
        return self.article.relevant_article

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "canonical_article": self.canonical_article,
            "article_title": self.article.article_title,
            "node_type": self.node_type,
            "norm_roles": self.norm_roles,
            "support_span": self.support_span,
            "support_snippet": self.support_snippet,
            "retrieval_trace": self.retrieval_trace,
        }


@dataclass(slots=True)
class EvidenceAnswer:
    answer: str
    used_evidence_ids: list[str]
    insufficient_evidence: bool = False
    support_map: list[dict[str, Any]] = field(default_factory=list)
    covered_components: list[str] = field(default_factory=list)
    insufficient_components: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_evidence_blocks(articles: list[ArticleNode], max_chars: int = 900) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    for idx, article in enumerate(articles, start=1):
        snippet = str(article.metadata.get("support_snippet") or compact_snippet(article.text, max_chars=max_chars))
        trace = str(article.metadata.get("retrieval_trace") or f"score={article.score:.4f}")
        blocks.append(
            EvidenceBlock(
                f"E{idx}",
                article,
                snippet,
                trace,
                node_type=str(article.metadata.get("node_type") or article.metadata.get("chunk_type") or "article"),
                norm_roles=[str(item) for item in article.metadata.get("norm_roles", []) if str(item).strip()],
                support_span=str(article.metadata.get("support_span_label") or article.metadata.get("support_span_key") or ""),
            )
        )
    return blocks


def generate_evidence_answer(
    question: str,
    plan: Any,
    evidence_blocks: list[EvidenceBlock],
    config: OllamaConfig,
) -> EvidenceAnswer:
    if not evidence_blocks:
        raise OllamaError("no_retrieved_articles")
    raw = request_ollama_chat(_answer_system_prompt(), _answer_user_prompt(question, plan, evidence_blocks), config)
    return evidence_answer_from_json(parse_json_response(raw, "answer"), evidence_blocks)


def evidence_answer_from_json(data: dict[str, Any], evidence_blocks: list[EvidenceBlock]) -> EvidenceAnswer:
    allowed = {"answer", "used_evidence_ids", "insufficient_evidence", "support_map", "covered_components", "insufficient_components"}
    extra = sorted(set(data) - allowed)
    if extra:
        raise OllamaError(f"answer_unexpected_keys:{','.join(extra)}")
    answer = str(data.get("answer", "")).strip()
    if not answer:
        raise OllamaError("answer_empty")

    available = {block.evidence_id for block in evidence_blocks}
    used = []
    for item in data.get("used_evidence_ids", []):
        value = str(item).strip()
        if value and value not in used:
            used.append(value)
    unknown = [item for item in used if item not in available]
    if unknown:
        raise OllamaError(f"answer_unknown_evidence_ids:{','.join(unknown)}")
    if not used and not bool(data.get("insufficient_evidence")):
        raise OllamaError("answer_missing_used_evidence_ids")

    support_map = data.get("support_map", [])
    if not isinstance(support_map, list):
        raise OllamaError("answer_support_map_not_list")
    return EvidenceAnswer(
        answer=answer,
        used_evidence_ids=used,
        insufficient_evidence=bool(data.get("insufficient_evidence")),
        support_map=[item for item in support_map if isinstance(item, dict)],
        covered_components=[str(item).strip() for item in data.get("covered_components", []) if str(item).strip()],
        insufficient_components=[str(item).strip() for item in data.get("insufficient_components", []) if str(item).strip()],
    )


def articles_from_used_evidence(evidence_blocks: list[EvidenceBlock], used_evidence_ids: list[str]) -> list[ArticleNode]:
    by_id = {block.evidence_id: block.article for block in evidence_blocks}
    output: list[ArticleNode] = []
    seen: set[str] = set()
    for evidence_id in used_evidence_ids:
        article = by_id.get(evidence_id)
        if article and article.article_key not in seen:
            seen.add(article.article_key)
            output.append(article)
    return output


def _answer_system_prompt() -> str:
    return (
        "Bạn là Legal Evidence Answerer cho pháp luật Việt Nam.\n\n"
        "Bạn chỉ được trả lời dựa trên EVIDENCE được cung cấp. Mỗi evidence có id E1, E2... gắn với đúng mã văn bản, "
        "tên văn bản và Điều. Không được dùng kiến thức ngoài evidence. Không được cite Điều hoặc văn bản không có trong evidence.\n\n"
        "Bạn phải chọn TỐI THIỂU evidence đủ để trả lời. Không đưa evidence dư vào used_evidence_ids. "
        "Nếu một evidence không trực tiếp hỗ trợ câu trả lời thì không dùng.\n\n"
        "Trả về JSON hợp lệ duy nhất, không markdown, không giải thích ngoài JSON."
    )


def _answer_user_prompt(question: str, plan: Any, evidence_blocks: list[EvidenceBlock]) -> str:
    plan_payload = _compact_plan_payload(plan)
    requested_components = []
    if hasattr(plan, "requested_components"):
        requested_components = getattr(plan, "requested_components") or []
    return (
        f"CÂU HỎI:\n{question}\n\n"
        f"QUERY PLAN:\n{plan_payload}\n\n"
        f"CÁC THÀNH PHẦN CẦN PHỦ NẾU CÓ CĂN CỨ:\n{requested_components}\n\n"
        "EVIDENCE:\n"
        + "\n\n".join(_format_block(block) for block in evidence_blocks)
        + """

Schema output:
{
  "answer": "...",
  "used_evidence_ids": ["E1"],
  "insufficient_evidence": false,
  "covered_components": ["..."],
  "insufficient_components": [],
  "support_map": [
    {
      "claim": "...",
      "evidence_ids": ["E1"],
      "article_refs": ["<doc_id>|<title>|<Điều X>"]
    }
  ]
}

Quy tắc:
- answer phải ngắn, trực tiếp, tiếng Việt.
- Nếu dùng căn cứ nào, answer phải nhắc rõ "Điều X"; nếu có nhiều văn bản có cùng Điều X thì nhắc thêm tên hoặc số hiệu văn bản.
- used_evidence_ids chỉ gồm evidence thật sự dùng trong answer.
- covered_components phải phản ánh các thành phần của câu trả lời đã được evidence hỗ trợ, như hồ sơ/cơ quan/thời hạn/mức phạt/biện pháp khắc phục.
- relevant_docs/relevant_articles sẽ được hệ thống lấy từ used_evidence_ids, nên không chọn dư.
- Nếu evidence chưa đủ, đặt insufficient_evidence=true và answer nói rõ chưa đủ căn cứ; không suy đoán.
- Không thêm key ngoài schema."""
    )


def _format_block(block: EvidenceBlock) -> str:
    return "\n".join(
        [
            block.evidence_id,
            f"canonical_article: {block.canonical_article}",
            f"article_title: {block.article.article_title}",
            f"node_type: {block.node_type}",
            f"norm_roles: {','.join(block.norm_roles)}",
            f"support_span: {block.support_span}",
            f"evidence_text: {_evidence_text(block)}",
            f"retrieval_trace: {block.retrieval_trace}",
        ]
    )


def _compact_plan_payload(plan: Any) -> dict[str, Any] | Any:
    if not hasattr(plan, "to_dict"):
        return plan
    payload = plan.to_dict()
    return {
        "intent": payload.get("intent"),
        "normalized_question": payload.get("normalized_question"),
        "question_type": payload.get("question_type"),
        "answer_shape": payload.get("answer_shape"),
        "legal_facets": payload.get("legal_facets", [])[:4],
        "requested_components": payload.get("requested_components", [])[:4],
        "target_norm_roles": payload.get("target_norm_roles", [])[:4],
        "governing_doc_hints": payload.get("governing_doc_hints", [])[:3],
        "domain_anchors": payload.get("domain_anchors", [])[:3],
        "needs_guidance_docs": payload.get("needs_guidance_docs", False),
    }


def _evidence_text(block: EvidenceBlock) -> str:
    if block.node_type in {"clause", "point", "micro_chunk"} and block.support_snippet:
        return compact_snippet(block.support_snippet, max_chars=900)
    return compact_snippet(block.article.text, max_chars=900)
