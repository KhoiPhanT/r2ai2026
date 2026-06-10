from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from legal_rag.corpus.ingest import iter_articles_jsonl
from legal_rag.schemas.models import ArticleNode


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
DOC_ID_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+(?:\d+)?\b")
STOPWORDS = {
    "của",
    "và",
    "về",
    "cho",
    "theo",
    "trong",
    "ngoài",
    "một",
    "các",
    "những",
    "được",
    "này",
    "đó",
    "khi",
    "với",
    "tại",
    "từ",
    "đến",
    "hoặc",
    "là",
    "có",
}
GENERIC_LEGAL_TOKENS = {
    "doanh nghiệp",
    "công ty",
    "nhân viên",
    "người",
    "hợp đồng",
    "quản lý",
    "quy định",
    "thực hiện",
    "hoạt động",
    "trách nhiệm",
    "hỗ trợ",
    "điều kiện",
}
LEGAL_HEAD_TERMS = (
    "hỗ trợ",
    "ưu đãi",
    "xử phạt",
    "vi phạm",
    "thẩm quyền",
    "thủ tục",
    "hồ sơ",
    "điều kiện",
    "thời hạn",
    "thời gian",
    "mức phạt",
    "khắc phục",
    "thuế",
    "hóa đơn",
    "đất đai",
    "mặt bằng",
    "thuê",
    "hợp đồng",
    "bảo hiểm",
    "sở hữu trí tuệ",
    "doanh nghiệp",
    "người lao động",
)
LAY_ALIASES = {
    "công ty": "doanh nghiệp",
    "nhân viên": "người lao động",
    "bằng cấp": "văn bằng chứng chỉ",
    "ký hợp đồng": "giao kết hợp đồng lao động",
    "cty": "doanh nghiệp",
    "vừa và nhỏ": "nhỏ và vừa",
    "giá thuê": "chi phí thuê",
    "bao lâu": "thời gian",
    "tối đa bao lâu": "thời gian tối đa",
}


@dataclass(slots=True)
class LegalLexiconEntry:
    term: str
    aliases: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    article_keys: list[str] = field(default_factory=list)
    norm_roles: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    regimes: list[str] = field(default_factory=list)
    source_spans: list[str] = field(default_factory=list)
    frequency: int = 0
    specificity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LegalLexiconEntry":
        return cls(
            term=str(data.get("term") or ""),
            aliases=[str(item) for item in data.get("aliases", []) if str(item).strip()],
            related_terms=[str(item) for item in data.get("related_terms", []) if str(item).strip()],
            doc_ids=[str(item) for item in data.get("doc_ids", []) if str(item).strip()],
            article_keys=[str(item) for item in data.get("article_keys", []) if str(item).strip()],
            norm_roles=[str(item) for item in data.get("norm_roles", []) if str(item).strip()],
            doc_types=[str(item) for item in data.get("doc_types", []) if str(item).strip()],
            regimes=[str(item) for item in data.get("regimes", []) if str(item).strip()],
            source_spans=[str(item) for item in data.get("source_spans", []) if str(item).strip()],
            frequency=int(data.get("frequency") or 0),
            specificity=float(data.get("specificity") or 0.0),
        )


@dataclass(slots=True)
class LegalLexiconMatch:
    term: str
    aliases: list[str]
    related_terms: list[str]
    doc_ids: list[str]
    article_keys: list[str]
    norm_roles: list[str]
    doc_types: list[str]
    regimes: list[str]
    source_spans: list[str]
    specificity: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_legal_query_text(text: str) -> str:
    lowered = " ".join(text.lower().split())
    for source, target in LAY_ALIASES.items():
        lowered = lowered.replace(source, target)
    return lowered


def build_legal_lexicon(articles_path: str | Path, output_path: str | Path, *, max_entries: int = 80_000) -> dict:
    stats: dict[str, dict] = {}
    neighbor_counts: dict[str, Counter[str]] = defaultdict(Counter)
    article_count = 0
    for article in iter_articles_jsonl(articles_path):
        article_count += 1
        term_sources, term_groups, defined_aliases = _article_term_data(article)
        terms = set(term_sources)
        if not terms:
            continue
        for term in terms:
            row = stats.setdefault(
                term,
                {
                    "term": term,
                    "aliases": set(),
                    "related_terms": set(),
                    "doc_ids": set(),
                    "article_keys": set(),
                    "norm_roles": set(),
                    "doc_types": set(),
                    "regimes": set(),
                    "source_spans": set(),
                    "frequency": 0,
                },
            )
            row["frequency"] += 1
            row["doc_ids"].add(article.doc_id)
            row["article_keys"].add(article.article_key)
            row["doc_types"].add(article.doc_type)
            row["regimes"].add(_document_regime(article))
            row["source_spans"].update(term_sources.get(term, [])[:4])
            for role in article.metadata.get("norm_roles", []) or []:
                row["norm_roles"].add(str(role))
            for alias in _aliases_for_term(term):
                row["aliases"].add(alias)
            row["aliases"].update(defined_aliases.get(term, []))
        for group in term_groups:
            group_terms = sorted(group, key=lambda value: (-len(value.split()), value))[:8]
            for left_index, left in enumerate(group_terms):
                for right in group_terms[left_index + 1 :]:
                    _increment_bounded_neighbor(neighbor_counts[left], right)
                    _increment_bounded_neighbor(neighbor_counts[right], left)

    for term, row in stats.items():
        for related, cooccurrence in neighbor_counts.get(term, Counter()).most_common(16):
            related_row = stats.get(related)
            if related_row is None or cooccurrence < 2:
                continue
            if not (row["regimes"] & related_row["regimes"]):
                continue
            pmi = math.log2(
                max(1.0, (cooccurrence * max(1, article_count)) / max(1, row["frequency"] * related_row["frequency"]))
            )
            if pmi >= 0.5:
                row["related_terms"].add(related)

    entries = [_entry_from_stats(row) for row in stats.values()]
    entries.sort(key=lambda item: (-item.frequency, -len(item.doc_ids), item.term))
    entries = entries[:max_entries]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix in {".sqlite", ".db"}:
        _write_lexicon_sqlite(entries, output)
        return {"articles": article_count, "entries": len(entries), "output": str(output), "backend": "sqlite_fts5"}
    with output.open("w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return {"articles": article_count, "entries": len(entries), "output": str(output)}


class LegalLexiconIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    def search(self, question: str, limit: int = 12) -> list[LegalLexiconMatch]:
        query = normalize_legal_query_text(question)
        tokens = [token for token in _tokens(query) if len(token) > 2][:10]
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
        rows = self.connection.execute(
            """
            SELECT e.payload, bm25(lexicon_fts, 5.0, 3.0, 1.0) AS rank
            FROM lexicon_fts
            JOIN entries e ON e.rowid = lexicon_fts.rowid
            WHERE lexicon_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (expression, max(limit * 8, 24)),
        ).fetchall()
        candidates = [LegalLexiconEntry.from_dict(json.loads(row["payload"])) for row in rows]
        return _rank_lexicon_entries(candidates, question, limit=limit)

    def close(self) -> None:
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
            self.connection = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load_legal_lexicon(path: str | Path | None) -> list[LegalLexiconEntry] | LegalLexiconIndex:
    if not path:
        return []
    lexicon_path = Path(path)
    if not lexicon_path.exists():
        return []
    if lexicon_path.suffix in {".sqlite", ".db"}:
        return LegalLexiconIndex(lexicon_path)
    entries: list[LegalLexiconEntry] = []
    with lexicon_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entries.append(LegalLexiconEntry.from_dict(json.loads(line)))
    return entries


def search_legal_lexicon(entries: Iterable[LegalLexiconEntry], question: str, *, limit: int = 12) -> list[LegalLexiconMatch]:
    if isinstance(entries, LegalLexiconIndex):
        return entries.search(question, limit=limit)
    return _rank_lexicon_entries(entries, question, limit=limit)


def _rank_lexicon_entries(entries: Iterable[LegalLexiconEntry], question: str, *, limit: int) -> list[LegalLexiconMatch]:
    query = normalize_legal_query_text(question)
    query_tokens = _tokens(query)
    matches: list[LegalLexiconMatch] = []
    for entry in entries:
        texts = [
            entry.term,
            *[alias for alias in entry.aliases if _useful_alias_for_entry(entry.term, alias)],
        ]
        score = _entry_score(query, query_tokens, texts, entry.frequency)
        score *= 0.55 + max(0.05, entry.specificity)
        if entry.term in LEGAL_HEAD_TERMS:
            score *= 0.35
        if score <= 0:
            continue
        matches.append(
            LegalLexiconMatch(
                term=entry.term,
                aliases=entry.aliases[:5],
                related_terms=entry.related_terms[:8],
                doc_ids=entry.doc_ids[:6],
                article_keys=entry.article_keys[:6],
                norm_roles=entry.norm_roles[:6],
                doc_types=entry.doc_types[:4],
                regimes=entry.regimes[:4],
                source_spans=entry.source_spans[:4],
                specificity=entry.specificity,
                score=round(score, 4),
            )
        )
    matches.sort(key=lambda item: (-item.score, item.term))
    diversified: list[LegalLexiconMatch] = []
    family_counts: dict[str, int] = {}
    for match in matches:
        family = _term_family(match.term)
        if family_counts.get(family, 0) >= 3:
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        diversified.append(match)
        if len(diversified) >= limit:
            break
    return diversified


def _article_term_sources(article: ArticleNode) -> dict[str, list[str]]:
    sources, _groups, _aliases = _article_term_data(article)
    return sources


def _article_term_data(article: ArticleNode) -> tuple[dict[str, list[str]], list[set[str]], dict[str, set[str]]]:
    output: dict[str, list[str]] = {}
    groups: list[set[str]] = []
    aliases: dict[str, set[str]] = defaultdict(set)
    segments = [
        article.title_for_submission,
        article.article_title,
        str(article.metadata.get("chapter_title") or ""),
        str(article.metadata.get("section_title") or ""),
    ]
    for clause in article.metadata.get("clause_nodes", []) or []:
        text = str(clause.get("text") or "").strip()
        if text:
            segments.append(text[:500])
    for sentence in _sentences(article.text[:2500]):
        sentence_terms = _known_legal_phrases(sentence)
        sentence_terms.update(term for term, _alias in _defined_alias_pairs(sentence))
        for term in sentence_terms:
            if _valid_term(term):
                output.setdefault(term, []).append(sentence[:300])
        valid_sentence_terms = {term for term in sentence_terms if _valid_term(term)}
        if len(valid_sentence_terms) >= 2:
            groups.append(valid_sentence_terms)
        for term, alias in _defined_alias_pairs(sentence):
            if _valid_term(term) and alias:
                output.setdefault(term, []).append(sentence[:300])
                aliases[term].add(alias)
    for segment in segments:
        if not segment:
            continue
        terms = _structured_terms(segment)
        valid_segment_terms: set[str] = set()
        for term in terms:
            if _valid_term(term):
                output.setdefault(term, []).append(segment[:300])
                valid_segment_terms.add(term)
        if len(valid_segment_terms) >= 2:
            groups.append(valid_segment_terms)
        for term, alias in _defined_alias_pairs(segment):
            if _valid_term(term) and alias:
                output.setdefault(term, []).append(segment[:300])
                aliases[term].add(alias)
    return output, groups, aliases


def _structured_terms(text: str) -> set[str]:
    terms = _known_legal_phrases(text)
    cleaned = _clean_phrase(text)
    if 2 <= len(cleaned.split()) <= 12 and any(head in cleaned for head in LEGAL_HEAD_TERMS):
        terms.add(cleaned)
    for match in re.finditer(r"(?i)([^.;:\n]{4,100}?)\s*\(sau đây gọi tắt là\s+([^\)]+)\)", text):
        full = _clean_phrase(match.group(1))
        if _valid_term(full):
            terms.add(full)
    return terms


def _defined_alias_pairs(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"(?i)([^.;:\n]{4,120}?)\s*\((?:sau đây gọi tắt là|viết tắt là)\s+[\"“]?([^\)\"”]+)[\"”]?\)"
    )
    output: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        term = _clean_phrase(match.group(1))
        alias = " ".join(match.group(2).split()).strip()
        if term and alias:
            output.append((term, alias))
    return output


def _increment_bounded_neighbor(counter: Counter[str], value: str, *, limit: int = 32) -> None:
    if value in counter or len(counter) < limit:
        counter[value] += 1


def _title_terms(text: str) -> set[str]:
    output: set[str] = set()
    normalized = _clean_phrase(text)
    if _valid_term(normalized):
        output.add(normalized)
    output.update(_known_legal_phrases(text))
    output.update(_known_legal_phrases(normalized))
    return output


def _known_legal_phrases(text: str) -> set[str]:
    phrases: set[str] = set()
    normalized = normalize_legal_query_text(text)
    if "doanh nghiệp nhỏ và vừa" in normalized:
        phrases.add("doanh nghiệp nhỏ và vừa")
    if "thuê mặt bằng" in normalized:
        phrases.add("thuê mặt bằng")
        if "hỗ trợ" in normalized:
            phrases.add("hỗ trợ thuê mặt bằng")
        if "chi phí thuê mặt bằng" in normalized:
            phrases.add("chi phí thuê mặt bằng")
    if "thời gian hỗ trợ tối đa" in normalized:
        phrases.add("thời gian hỗ trợ tối đa")
    elif "thời gian hỗ trợ" in normalized:
        phrases.add("thời gian hỗ trợ")
    if "cơ sở ươm tạo" in normalized:
        phrases.add("cơ sở ươm tạo")
    if "khu làm việc chung" in normalized:
        phrases.add("khu làm việc chung")
    if "xử phạt" in normalized and "hóa đơn" in normalized:
        phrases.add("xử phạt hóa đơn")
    if "thẩm quyền" in normalized and "xử phạt" in normalized:
        phrases.add("thẩm quyền xử phạt")
    for doc_id in DOC_ID_RE.findall(text):
        phrases.add(doc_id)
    return phrases


def _window_terms(sentence: str) -> set[str]:
    words = _tokens(sentence)
    output: set[str] = set()
    if not words:
        return output
    for head in LEGAL_HEAD_TERMS:
        head_words = head.split()
        size = len(head_words)
        for idx in range(0, len(words) - size + 1):
            if words[idx : idx + size] != head_words:
                continue
            forward_end = min(len(words), idx + size + 5)
            backward_start = max(0, idx - 2)
            output.add(_clean_phrase(" ".join(words[idx:forward_end])))
            output.add(_clean_phrase(" ".join(words[backward_start:forward_end])))
            output.add(head)
            if len(output) >= 24:
                return output
    return output


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\n.;:!?]+", text) if part.strip()]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(normalize_legal_query_text(text)) if token.lower() not in STOPWORDS]


def _clean_phrase(text: str) -> str:
    words = _tokens(text)
    while words and words[0] in STOPWORDS:
        words.pop(0)
    while words and words[-1] in STOPWORDS:
        words.pop()
    return " ".join(words[:12])


def _valid_term(term: str) -> bool:
    if not term or len(term) < 5:
        return False
    words = term.split()
    if len(words) < 2 or len(words) > 12:
        return False
    if words[0].isdigit() or (len(words[0]) == 1 and words[0].isalpha()):
        return False
    if all(word in STOPWORDS for word in words):
        return False
    return any(head in term for head in LEGAL_HEAD_TERMS) or bool(DOC_ID_RE.search(term))


def _aliases_for_term(term: str) -> list[str]:
    aliases = {term}
    if "doanh nghiệp nhỏ và vừa" in term:
        aliases.update({"công ty nhỏ và vừa", "doanh nghiệp vừa và nhỏ", "sme", "dnnvv"})
    if "thuê mặt bằng" in term:
        aliases.update({"giá thuê mặt bằng", "chi phí thuê mặt bằng", "mặt bằng sản xuất"})
    if "thời gian hỗ trợ" in term:
        aliases.update({"thời gian tối đa", "tối đa bao lâu", "bao lâu"})
    return sorted(aliases - {term})


def _term_family(term: str) -> str:
    lowered = term.lower()
    if "doanh nghiệp nhỏ" in lowered:
        return "sme"
    if "thuê mặt bằng" in lowered or "mặt bằng" in lowered:
        return "rental"
    if "thời gian" in lowered or "thời hạn" in lowered:
        return "time"
    if "xử phạt" in lowered or "mức phạt" in lowered:
        return "penalty"
    if "hồ sơ" in lowered or "thủ tục" in lowered:
        return "procedure"
    tokens = lowered.split()
    return tokens[0] if tokens else lowered


def _entry_score(query: str, query_tokens: list[str], texts: list[str], frequency: int) -> float:
    best = 0.0
    query_set = set(query_tokens)
    for text in texts:
        candidate = normalize_legal_query_text(text)
        if not candidate:
            continue
        tokens = set(_tokens(candidate))
        if not tokens:
            continue
        useful_tokens = {token for token in tokens if token not in _generic_token_parts()}
        if candidate in query:
            if not _useful_lookup_phrase(candidate):
                continue
            best = max(best, 4.0 + min(len(tokens), 8) * 0.18 + min(frequency, 20) * 0.01)
            if any(marker in candidate or marker in text.lower() for marker in ("doanh nghiệp nhỏ và vừa", "thời gian hỗ trợ", "thuê mặt bằng")):
                best += 1.0
            if any(marker in candidate or marker in text.lower() for marker in ("thời gian hỗ trợ", "thời gian tối đa", "bao lâu")):
                best += 1.5
        overlap = len(query_set & tokens)
        useful_overlap = len(query_set & useful_tokens)
        if overlap >= min(2, len(tokens)) and useful_overlap >= 1 and overlap / max(1, len(tokens)) >= 0.4:
            precision = overlap / max(1, len(tokens))
            recall = overlap / max(1, len(query_set))
            best = max(best, 2.0 * precision + 1.5 * recall + min(frequency, 20) * 0.01)
    return best


def _useful_lookup_phrase(value: str) -> bool:
    normalized = normalize_legal_query_text(value)
    tokens = set(_tokens(normalized))
    if len(tokens) < 2:
        return False
    return normalized not in GENERIC_LEGAL_TOKENS and bool(tokens - _generic_token_parts())


def _useful_alias_for_entry(term: str, alias: str) -> bool:
    if not _useful_lookup_phrase(alias):
        return False
    term_tokens = set(_tokens(term))
    alias_tokens = set(_tokens(alias))
    if len(term_tokens) <= 7:
        return True
    return len(term_tokens & alias_tokens) / max(1, len(term_tokens)) >= 0.5


def _generic_token_parts() -> set[str]:
    return {token for phrase in GENERIC_LEGAL_TOKENS for token in phrase.split()}


def _entry_from_stats(row: dict) -> LegalLexiconEntry:
    frequency = int(row["frequency"])
    doc_count = len(row["doc_ids"])
    specificity = min(1.0, max(0.05, (len(row["term"].split()) / 8.0) + (1.0 / max(2, doc_count))))
    return LegalLexiconEntry(
        term=row["term"],
        aliases=sorted(row["aliases"])[:12],
        related_terms=sorted(row["related_terms"], key=lambda item: (len(item), item))[:16],
        doc_ids=sorted(row["doc_ids"])[:16],
        article_keys=sorted(row["article_keys"])[:24],
        norm_roles=sorted(row["norm_roles"]),
        doc_types=sorted(row["doc_types"]),
        regimes=sorted(row["regimes"]),
        source_spans=sorted(row["source_spans"], key=lambda item: (len(item), item))[:8],
        frequency=frequency,
        specificity=round(specificity, 4),
    )


def _document_regime(article: ArticleNode) -> str:
    title = " ".join(article.title_for_submission.split())
    return title[:180]


def _write_lexicon_sqlite(entries: list[LegalLexiconEntry], output: Path) -> None:
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    connection = sqlite3.connect(tmp)
    try:
        connection.executescript(
            """
            CREATE TABLE entries (rowid INTEGER PRIMARY KEY, term TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE VIRTUAL TABLE lexicon_fts USING fts5(
                term,
                aliases,
                regimes,
                content='',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        for rowid, entry in enumerate(entries, start=1):
            payload = json.dumps(entry.to_dict(), ensure_ascii=False)
            connection.execute("INSERT INTO entries(rowid, term, payload) VALUES (?, ?, ?)", (rowid, entry.term, payload))
            connection.execute(
                "INSERT INTO lexicon_fts(rowid, term, aliases, regimes) VALUES (?, ?, ?, ?)",
                (rowid, entry.term, " ".join(entry.aliases), " ".join(entry.regimes)),
            )
        connection.execute("INSERT INTO index_meta(key, value) VALUES ('schema_version', '2')")
        connection.execute("INSERT INTO index_meta(key, value) VALUES ('entries', ?)", (str(len(entries)),))
        connection.commit()
        connection.execute("INSERT INTO lexicon_fts(lexicon_fts) VALUES ('optimize')")
        connection.commit()
    finally:
        connection.close()
    tmp.replace(output)
