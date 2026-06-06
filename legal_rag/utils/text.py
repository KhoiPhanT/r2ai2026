from __future__ import annotations

import re
import unicodedata

ARTICLE_PATTERN = re.compile(r"\bĐiều\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
ARTICLE_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(Điều\s+\d+[a-zA-Z]?)\s*[\.:]?\s*(.*)$"
)
DOC_ID_PATTERN = re.compile(
    r"\b\d{1,4}/\d{4}/[A-ZĐÂÊÔƠƯa-zđâêôơư0-9.-]+\b", re.UNICODE
)
TOKEN_PATTERN = re.compile(r"[\w/.-]+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[“”\"'`]", "", text)
    return text


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(normalize_for_match(text))]


def extract_article_labels(text: str) -> list[str]:
    seen: set[str] = set()
    labels: list[str] = []
    for match in ARTICLE_PATTERN.finditer(text or ""):
        label = f"Điều {match.group(1)}"
        key = label.lower()
        if key not in seen:
            seen.add(key)
            labels.append(label)
    return labels


def extract_doc_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for match in DOC_ID_PATTERN.finditer(text or ""):
        doc_id = match.group(0)
        key = doc_id.upper()
        if key not in seen:
            seen.add(key)
            ids.append(doc_id)
    return ids


def compact_snippet(text: str, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", normalize_text(text))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."

