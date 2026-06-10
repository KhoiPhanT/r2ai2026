from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from legal_rag.generation.ollama import OllamaConfig, OllamaError, parse_json_response, request_ollama_chat
from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import extract_article_labels
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
    claims: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_evidence_blocks(articles: list[ArticleNode], max_chars: int = 900) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    seen: set[str] = set()
    for article in articles:
        if article.article_key in seen:
            continue
        seen.add(article.article_key)
        snippet = str(article.metadata.get("support_snippet") or compact_snippet(article.text, max_chars=max_chars))
        trace = str(article.metadata.get("retrieval_trace") or f"score={article.score:.4f}")
        blocks.append(
            EvidenceBlock(
                f"E{len(blocks) + 1}",
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
    runtime_config = answer_runtime_config(plan, config)
    raw = request_ollama_chat(
        _answer_system_prompt(),
        _answer_user_prompt(question, plan, evidence_blocks),
        runtime_config,
        json_schema=_answer_json_schema(),
    )
    try:
        return evidence_answer_from_json(parse_json_response(raw, "answer"), evidence_blocks)
    except OllamaError:
        repair_config = replace(
            runtime_config,
            think=True,
            num_ctx=max(runtime_config.num_ctx, 12288),
            max_tokens=max(runtime_config.max_tokens, 650),
        )
        repaired = request_ollama_chat(
            "Bạn sửa output Legal Evidence Answerer thành JSON đúng schema. Không thêm kiến thức hoặc evidence mới.",
            (
                f"CÂU HỎI:\n{question}\n\n"
                f"EVIDENCE IDS HỢP LỆ: {[block.evidence_id for block in evidence_blocks]}\n\n"
                f"OUTPUT CẦN SỬA:\n{raw[:6000]}\n\n"
                "Nếu không đủ căn cứ, đặt insufficient_evidence=true. Trả duy nhất JSON."
            ),
            repair_config,
            json_schema=_answer_json_schema(),
        )
        return evidence_answer_from_json(parse_json_response(repaired, "answer_repair"), evidence_blocks)


def repair_evidence_answer(
    question: str,
    plan: Any,
    evidence_blocks: list[EvidenceBlock],
    previous: EvidenceAnswer,
    missing_components: list[str],
    config: OllamaConfig,
) -> EvidenceAnswer:
    repair_config = replace(
        config,
        think=True,
        num_ctx=max(config.num_ctx, 16384),
        max_tokens=max(config.max_tokens, 700),
    )
    user_prompt = (
        _answer_user_prompt(question, plan, evidence_blocks)
        + "\n\nBẢN TRƯỚC CẦN SỬA:\n"
        + str(previous.to_dict())
        + "\n\nTHÀNH PHẦN BẮT BUỘC CÒN THIẾU:\n"
        + ", ".join(missing_components)
        + "\nChỉ bổ sung claim có evidence trực tiếp. Nếu evidence không đủ, ghi đúng thành phần vào insufficient_components."
    )
    raw = request_ollama_chat(
        _answer_system_prompt(),
        user_prompt,
        repair_config,
        json_schema=_answer_json_schema(),
    )
    return evidence_answer_from_json(parse_json_response(raw, "answer_repair"), evidence_blocks)


def evidence_answer_from_json(data: dict[str, Any], evidence_blocks: list[EvidenceBlock]) -> EvidenceAnswer:
    allowed = {
        "answer",
        "answer_text",
        "used_evidence_ids",
        "insufficient_evidence",
        "support_map",
        "claims",
        "covered_components",
        "insufficient_components",
    }
    extra = sorted(set(data) - allowed)
    if extra:
        raise OllamaError(f"answer_unexpected_keys:{','.join(extra)}")
    answer = str(data.get("answer_text") or data.get("answer") or "").strip()
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

    support_map = data.get("claims") or data.get("support_map", [])
    if not isinstance(support_map, list):
        raise OllamaError("answer_support_map_not_list")
    support_map = [item for item in support_map if isinstance(item, dict)]
    used = _minimize_used_evidence(answer, used, support_map, evidence_blocks, bool(data.get("insufficient_evidence")))
    return EvidenceAnswer(
        answer=answer,
        used_evidence_ids=used,
        insufficient_evidence=bool(data.get("insufficient_evidence")),
        support_map=support_map,
        covered_components=[str(item).strip() for item in data.get("covered_components", []) if str(item).strip()],
        insufficient_components=[str(item).strip() for item in data.get("insufficient_components", []) if str(item).strip()],
        claims=support_map,
    )


def answer_runtime_config(plan: Any, config: OllamaConfig) -> OllamaConfig:
    question_type = str(getattr(plan, "question_type", ""))
    answer_shape = str(getattr(plan, "answer_shape", "single_rule"))
    required = list(getattr(plan, "requested_components", []) or [])
    hard_case = question_type == "comparison" or len(required) >= 3
    if hard_case:
        return replace(
            config,
            think=True,
            num_ctx=max(config.num_ctx, 16384),
            max_tokens=max(config.max_tokens, 700),
        )
    if answer_shape in {"list_items", "procedure_steps", "conditions_list"}:
        return replace(config, max_tokens=min(config.max_tokens, 520))
    if answer_shape == "penalty_and_remedy":
        return replace(config, max_tokens=min(config.max_tokens, 420))
    return replace(
        config,
        max_tokens=min(config.max_tokens, 360),
    )


def _answer_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer_text": {"type": "string"},
            "used_evidence_ids": {"type": "array", "items": {"type": "string"}},
            "insufficient_evidence": {"type": "boolean"},
            "covered_components": {"type": "array", "items": {"type": "string"}},
            "insufficient_components": {"type": "array", "items": {"type": "string"}},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "article_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["claim", "evidence_ids"],
                },
            },
        },
        "required": ["answer_text", "used_evidence_ids", "insufficient_evidence", "claims"],
        "additionalProperties": False,
    }


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


def finalize_evidence_answer(evidence_answer: EvidenceAnswer, evidence_blocks: list[EvidenceBlock]) -> list[ArticleNode]:
    supported_ids = _supported_evidence_ids(evidence_answer.claims or evidence_answer.support_map, evidence_blocks)
    if supported_ids:
        evidence_answer.used_evidence_ids = supported_ids
    used_articles = articles_from_used_evidence(evidence_blocks, evidence_answer.used_evidence_ids)
    evidence_answer.answer = _render_canonical_citations(evidence_answer.answer, used_articles)
    return used_articles


def _answer_system_prompt() -> str:
    return (
        "Bạn là Legal Evidence Answerer cho pháp luật Việt Nam.\n\n"
        "Bạn chỉ được trả lời dựa trên EVIDENCE được cung cấp. Mỗi evidence có id E1, E2... gắn với đúng mã văn bản, "
        "tên văn bản và Điều. Không được dùng kiến thức ngoài evidence. Không được cite Điều hoặc văn bản không có trong evidence.\n\n"
        "Bạn phải chọn TỐI THIỂU evidence đủ để trả lời. Không đưa evidence dư vào used_evidence_ids. "
        "Nếu một evidence không trực tiếp hỗ trợ một claim trong answer thì không dùng.\n\n"
        "Không tự viết citation hoặc cụm 'Căn cứ Điều...'; hệ thống sẽ dựng citation từ evidence_ids.\n\n"
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
  "answer_text": "...",
  "used_evidence_ids": ["E1"],
  "insufficient_evidence": false,
  "covered_components": ["..."],
  "insufficient_components": [],
  "claims": [
    {
      "claim": "...",
      "evidence_ids": ["E1"],
      "article_refs": ["<doc_id>|<title>|<Điều X>"]
    }
  ]
}

Quy tắc:
- answer_text phải ngắn, trực tiếp, tiếng Việt và không tự ghi citation/Điều luật.
- Nếu câu hỏi hỏi "gồm những gì"/danh sách/hồ sơ và evidence_text có các điểm a), b), c) trong cùng support_span, phải liệt kê đủ các điểm trực tiếp liên quan; không chọn một mục đơn lẻ rồi bỏ các mục còn lại.
- used_evidence_ids chỉ gồm evidence thật sự dùng trong answer.
- claims là căn cứ kiểm tra used_evidence_ids: evidence nào không xuất hiện trong claims thì không đưa vào used_evidence_ids.
- covered_components phải phản ánh các thành phần của câu trả lời đã được evidence hỗ trợ, như hồ sơ/cơ quan/thời hạn/mức phạt/biện pháp khắc phục.
- Với câu hỏi thẩm quyền/cơ quan, chỉ coi evidence là đủ nếu nêu trực tiếp chủ thể có thẩm quyền. Nếu evidence chỉ nói "thực hiện theo quy định của pháp luật..." hoặc dẫn sang văn bản khác, phải đánh insufficient cho thẩm quyền/cơ quan; không tự suy ra tên cơ quan.
- Nếu một thành phần trong CÁC THÀNH PHẦN CẦN PHỦ không có evidence trực tiếp, không được bỏ qua âm thầm: nêu rõ trong answer là chưa tìm thấy/không đủ căn cứ cho thành phần đó và đưa đúng tên thành phần vào insufficient_components.
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
    if block.support_snippet:
        return compact_snippet(block.support_snippet, max_chars=1400)
    return compact_snippet(block.article.text, max_chars=1200)


def _minimize_used_evidence(
    answer: str,
    used: list[str],
    support_map: list[dict[str, Any]],
    evidence_blocks: list[EvidenceBlock],
    insufficient: bool,
) -> list[str]:
    if insufficient and not used:
        return []

    available = {block.evidence_id for block in evidence_blocks}
    support_ids: list[str] = []
    for item in support_map:
        raw_ids = item.get("evidence_ids", [])
        if not isinstance(raw_ids, list):
            continue
        for raw_id in raw_ids:
            evidence_id = str(raw_id).strip()
            if evidence_id in available and evidence_id not in support_ids:
                support_ids.append(evidence_id)

    if support_ids:
        base = [evidence_id for evidence_id in used if evidence_id in support_ids]
        for evidence_id in support_ids:
            if evidence_id not in base:
                base.append(evidence_id)
    else:
        base = used[:]

    cited_labels = {label.lower() for label in extract_article_labels(answer)}
    if cited_labels and base:
        by_id = {block.evidence_id: block for block in evidence_blocks}
        cited_base = [
            evidence_id
            for evidence_id in base
            if by_id.get(evidence_id) and by_id[evidence_id].article.article_label.lower() in cited_labels
        ]
        if cited_base:
            base = cited_base

    output: list[str] = []
    for evidence_id in base:
        if evidence_id in available and evidence_id not in output:
            output.append(evidence_id)
    return output


def _supported_evidence_ids(claims: list[dict[str, Any]], evidence_blocks: list[EvidenceBlock]) -> list[str]:
    available = {block.evidence_id for block in evidence_blocks}
    output: list[str] = []
    for claim in claims:
        for raw_id in claim.get("evidence_ids", []) if isinstance(claim, dict) else []:
            evidence_id = str(raw_id).strip()
            if evidence_id in available and evidence_id not in output:
                output.append(evidence_id)
    return output


def _render_canonical_citations(answer: str, articles: list[ArticleNode]) -> str:
    cleaned = re.sub(r"(?i)\s*Căn cứ\s+Điều\s+\d+[A-Za-z]?(?:[^.!?]*[.!?])?", " ", answer).strip()
    cleaned = re.sub(r"(?i)\s+theo\s+(?:quy định\s+tại\s+)?Điều\s+\d+[A-Za-z]?(?:\s+[^,.!?]+)?", "", cleaned).strip()
    cleaned = " ".join(cleaned.split())
    if not articles:
        return cleaned
    citations = []
    for article in articles:
        citation = f"{article.article_label} {article.doc_id}"
        if citation not in citations:
            citations.append(citation)
    suffix = "Căn cứ " + "; ".join(citations) + "."
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return f"{cleaned} {suffix}".strip()
