from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import compact_snippet

DEFAULT_OLLAMA_MODEL = "qwen3:8b-q8_0"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"


class OllamaError(RuntimeError):
    pass


@dataclass(slots=True)
class OllamaConfig:
    model: str = DEFAULT_OLLAMA_MODEL
    url: str = DEFAULT_OLLAMA_URL
    max_tokens: int = 700
    temperature: float = 0.0
    timeout: float = 600.0
    response_format_json: bool = False


def generate_ollama_answer(question: str, articles: list[ArticleNode], config: OllamaConfig) -> str:
    if not articles:
        raise OllamaError("no_retrieved_articles")

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(question, articles)},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "reasoning_effort": "none",
    }
    if config.response_format_json:
        payload["response_format"] = {"type": "json_object"}
    request = Request(
        config.url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"ollama_http_error:{exc.code}:{body}") from exc
    except URLError as exc:
        raise OllamaError(f"ollama_connection_error:{exc.reason}") from exc
    except TimeoutError as exc:
        raise OllamaError("ollama_timeout") from exc

    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"].get("content", "")
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"ollama_invalid_response:{raw[:500]}") from exc

    answer = _strip_thinking(str(content)).strip()
    if not answer:
        raise OllamaError("ollama_empty_answer")
    return answer


def request_ollama_chat(system_prompt: str, user_prompt: str, config: OllamaConfig) -> str:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "reasoning_effort": "none",
    }
    if config.response_format_json:
        payload["response_format"] = {"type": "json_object"}
    request = Request(
        config.url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"ollama_http_error:{exc.code}:{body}") from exc
    except URLError as exc:
        raise OllamaError(f"ollama_connection_error:{exc.reason}") from exc
    except TimeoutError as exc:
        raise OllamaError("ollama_timeout") from exc

    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"].get("content", "")
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"ollama_invalid_response:{raw[:500]}") from exc
    content = _strip_thinking(str(content)).strip()
    if not content:
        raise OllamaError("ollama_empty_answer")
    return content


def parse_json_response(content: str, label: str) -> dict:
    text = _strip_thinking(content).strip()
    fence = re.match(r"(?is)^```(?:json)?\s*(.*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"{label}_invalid_json:{exc.msg}") from exc
    if not isinstance(data, dict):
        raise OllamaError(f"{label}_json_not_object")
    return data


def _system_prompt() -> str:
    return (
        "Bạn là hệ thống Legal RAG cho pháp luật Việt Nam. "
        "Chỉ trả lời dựa trên các CĂN CỨ được cung cấp. "
        "Không tự thêm văn bản, số hiệu, hoặc Điều luật ngoài danh sách căn cứ. "
        "Trong câu trả lời, nếu nêu căn cứ thì phải nhắc đúng nhãn Điều xuất hiện ở đầu mỗi căn cứ. "
        "Không viết quá trình suy luận nội bộ."
    )


def _user_prompt(question: str, articles: list[ArticleNode]) -> str:
    evidence = []
    for idx, article in enumerate(articles, start=1):
        evidence.append(
            "\n".join(
                [
                    f"[{idx}] {article.relevant_article}",
                    f"Tên văn bản: {article.title_for_submission}",
                    f"Nội dung: {compact_snippet(article.text, max_chars=1600)}",
                ]
            )
        )
    return (
        f"CÂU HỎI:\n{question}\n\n"
        "CĂN CỨ ĐƯỢC TRUY HỒI:\n"
        + "\n\n".join(evidence)
        + "\n\nYÊU CẦU TRẢ LỜI:\n"
        "- Trả lời ngắn gọn, trực tiếp bằng tiếng Việt.\n"
        "- Chỉ dùng các căn cứ ở trên.\n"
        "- Phải nhắc rõ các nhãn như Điều 4, Điều 5 nếu dùng làm căn cứ.\n"
        "- Nếu căn cứ chưa đủ, nói rõ chưa đủ căn cứ và không suy đoán."
    )


def _strip_thinking(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    return text.replace("<think>", "").replace("</think>", "").strip()
