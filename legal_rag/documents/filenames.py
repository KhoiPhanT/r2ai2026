from __future__ import annotations

import re
import unicodedata

MAX_FILENAME_STEM = 160

_NON_WORD = re.compile(r"[^a-zA-Z0-9]+")
_MULTI_DASH = re.compile(r"-+")


def slug_doc_id(doc_id: str) -> str:
    return _MULTI_DASH.sub("-", doc_id.replace("/", "-").replace("_", "-")).strip("-")


def slug_title(title: str, max_len: int = 120) -> str:
    text = (title or "").replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = _NON_WORD.sub("-", ascii_text.lower())
    slug = _MULTI_DASH.sub("-", slug).strip("-")
    return slug[:max_len].strip("-") or "van-ban"


def normalized_docx_name(doc_id: str, title: str, duplicate_index: int = 0) -> str:
    doc_part = slug_doc_id(doc_id)
    title_part = slug_title(title)
    suffix = f"__dup{duplicate_index}" if duplicate_index else ""
    stem = f"{doc_part}__{title_part}{suffix}"
    if len(stem) > MAX_FILENAME_STEM:
        available = MAX_FILENAME_STEM - len(doc_part) - len(suffix) - 2
        title_part = title_part[: max(24, available)].strip("-")
        stem = f"{doc_part}__{title_part}{suffix}"
    return f"{stem}.docx"
