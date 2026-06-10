from __future__ import annotations

import json
import re
import sqlite3
import zlib
from pathlib import Path
from typing import Iterable

from legal_rag.corpus.ingest import iter_articles_jsonl
from legal_rag.schemas.models import ArticleNode


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


class FTS5Index:
    backend_name = "fts5_bm25"

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.last_trace: dict = {}

    @classmethod
    def load(cls, path: str | Path) -> "FTS5Index":
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return cls(path)

    def search(self, query: str, top_k: int = 20) -> list[ArticleNode]:
        expression = _fts_expression(query)
        if not expression or top_k <= 0:
            self.last_trace = {"candidate_counts": {"fts": 0}, "actual_backend": self.backend_name}
            return []
        rows = self.connection.execute(
            """
            SELECT a.payload, bm25(article_fts, 8.0, 4.0, 6.0, 3.0, 1.0) AS rank
            FROM article_fts
            JOIN articles a ON a.rowid = article_fts.rowid
            WHERE article_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (expression, int(top_k)),
        ).fetchall()
        output = []
        for row in rows:
            article = _decode_article(row["payload"])
            article.score = -float(row["rank"])
            output.append(article)
        self.last_trace = {
            "candidate_counts": {"fts": len(output)},
            "fused": len(output),
            "reranked": len(output),
            "final": len(output),
            "actual_backend": self.backend_name,
        }
        return output

    def exact_search(self, text: str, top_k: int = 20) -> list[ArticleNode]:
        doc_ids = _doc_ids(text)
        labels = _article_labels(text)
        clauses: list[str] = []
        params: list[str | int] = []
        if doc_ids:
            clauses.append(f"upper(doc_id) IN ({','.join('?' for _ in doc_ids)})")
            params.extend(item.upper() for item in doc_ids)
        if labels:
            clauses.append(f"article_label IN ({','.join('?' for _ in labels)})")
            params.extend(labels)
        if not clauses:
            return []
        params.append(int(top_k))
        rows = self.connection.execute(
            f"SELECT payload FROM articles WHERE {' AND '.join(clauses)} LIMIT ?",
            params,
        ).fetchall()
        output = [_decode_article(row["payload"]) for row in rows]
        for index, article in enumerate(output):
            article.score = 1000.0 - index
        return output

    def get_article(self, article_key: str) -> ArticleNode | None:
        row = self.connection.execute("SELECT payload FROM articles WHERE article_key = ?", (article_key,)).fetchone()
        return _decode_article(row["payload"]) if row else None

    def articles_for_doc_ids(self, doc_ids: set[str], limit: int = 100) -> list[ArticleNode]:
        if not doc_ids:
            return []
        ordered = sorted(doc_ids)
        rows = self.connection.execute(
            f"SELECT payload FROM articles WHERE doc_id IN ({','.join('?' for _ in ordered)}) LIMIT ?",
            (*ordered, int(limit)),
        ).fetchall()
        return [_decode_article(row["payload"]) for row in rows]

    def related_doc_ids(self, doc_ids: set[str]) -> set[str]:
        if not doc_ids:
            return set()
        ordered = sorted(doc_ids)
        placeholders = ",".join("?" for _ in ordered)
        rows = self.connection.execute(
            f"""
            SELECT source_doc_id, target_doc_id FROM relations
            WHERE source_doc_id IN ({placeholders}) OR target_doc_id IN ({placeholders})
            """,
            (*ordered, *ordered),
        ).fetchall()
        related: set[str] = set()
        for row in rows:
            source = str(row["source_doc_id"])
            target = str(row["target_doc_id"])
            if source in doc_ids:
                related.add(target)
            if target in doc_ids:
                related.add(source)
        return related - doc_ids

    def superseded_doc_ids(self, candidate_doc_ids: set[str]) -> set[str]:
        """Return candidates explicitly made ineffective by another candidate document."""
        if len(candidate_doc_ids) < 2:
            return set()
        ordered = sorted(candidate_doc_ids)
        placeholders = ",".join("?" for _ in ordered)
        rows = self.connection.execute(
            f"""
            SELECT DISTINCT target_doc_id FROM relations
            WHERE relation_type IN ('replaces_doc_ids', 'abolishes_doc_ids')
              AND source_doc_id IN ({placeholders})
              AND target_doc_id IN ({placeholders})
            """,
            (*ordered, *ordered),
        ).fetchall()
        return {str(row["target_doc_id"]) for row in rows}

    def superseding_sources(self, target_doc_ids: set[str]) -> dict[str, set[str]]:
        if not target_doc_ids:
            return {}
        ordered = sorted(target_doc_ids)
        placeholders = ",".join("?" for _ in ordered)
        rows = self.connection.execute(
            f"""
            SELECT DISTINCT source_doc_id, target_doc_id FROM relations
            WHERE relation_type IN ('replaces_doc_ids', 'abolishes_doc_ids')
              AND target_doc_id IN ({placeholders})
            """,
            ordered,
        ).fetchall()
        output: dict[str, set[str]] = {}
        for row in rows:
            output.setdefault(str(row["target_doc_id"]), set()).add(str(row["source_doc_id"]))
        return output

    def close(self) -> None:
        self.connection.close()


def build_fts5_index(articles_path: str | Path, output_path: str | Path) -> dict:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    connection = sqlite3.connect(tmp)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(
            """
            CREATE TABLE articles (
                rowid INTEGER PRIMARY KEY,
                article_key TEXT NOT NULL UNIQUE,
                doc_id TEXT NOT NULL,
                article_label TEXT NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE INDEX articles_doc_id ON articles(doc_id);
            CREATE INDEX articles_label ON articles(article_label);
            CREATE TABLE relations (
                source_doc_id TEXT NOT NULL,
                target_doc_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                UNIQUE(source_doc_id, target_doc_id, relation_type)
            );
            CREATE INDEX relations_source ON relations(source_doc_id);
            CREATE INDEX relations_target ON relations(target_doc_id);
            CREATE VIRTUAL TABLE article_fts USING fts5(
                doc_id,
                title,
                article_label,
                article_title,
                body,
                content='',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        count = 0
        for count, article in enumerate(iter_articles_jsonl(articles_path), start=1):
            payload = zlib.compress(json.dumps(article.to_dict(), ensure_ascii=False).encode("utf-8"), level=6)
            connection.execute(
                "INSERT INTO articles(rowid, article_key, doc_id, article_label, payload) VALUES (?, ?, ?, ?, ?)",
                (count, article.article_key, article.doc_id, article.article_label, payload),
            )
            relations = article.metadata.get("document_relations", {}) or {}
            for relation_type in ("guides_doc_ids", "amends_doc_ids", "replaces_doc_ids", "abolishes_doc_ids"):
                for target in relations.get(relation_type, []) or []:
                    target_doc_id = str(target).strip()
                    if target_doc_id and target_doc_id != article.doc_id:
                        connection.execute(
                            "INSERT OR IGNORE INTO relations(source_doc_id, target_doc_id, relation_type) VALUES (?, ?, ?)",
                            (article.doc_id, target_doc_id, relation_type),
                        )
            for relation_type, target_doc_id in _explicit_status_relations(article):
                connection.execute(
                    "INSERT OR IGNORE INTO relations(source_doc_id, target_doc_id, relation_type) VALUES (?, ?, ?)",
                    (article.doc_id, target_doc_id, relation_type),
                )
            connection.execute(
                "INSERT INTO article_fts(rowid, doc_id, title, article_label, article_title, body) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    count,
                    article.doc_id,
                    article.title_for_submission,
                    article.article_label,
                    article.article_title,
                    article.text,
                ),
            )
            if count % 2000 == 0:
                connection.commit()
        connection.execute("INSERT INTO index_meta(key, value) VALUES ('articles', ?)", (str(count),))
        connection.execute("INSERT INTO index_meta(key, value) VALUES ('schema_version', '2')")
        connection.commit()
        connection.execute("INSERT INTO article_fts(article_fts) VALUES ('optimize')")
        connection.commit()
    finally:
        connection.close()
    tmp.replace(output)
    return {"articles": count, "output": str(output), "backend": "fts5_bm25"}


def _decode_article(payload: bytes) -> ArticleNode:
    return ArticleNode.from_dict(json.loads(zlib.decompress(payload).decode("utf-8")))


def _fts_expression(query: str) -> str:
    tokens = []
    seen = set()
    for token in TOKEN_RE.findall(query.lower()):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token.replace('"', ''))
    if not tokens:
        return ""
    strong = tokens[:12]
    phrase = '"' + " ".join(strong[:6]) + '"' if len(strong) >= 2 else ""
    terms = " OR ".join(f'"{token}"' for token in strong)
    return f"{phrase} OR {terms}" if phrase else terms


def _doc_ids(text: str) -> list[str]:
    return re.findall(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+(?:\d+)?\b", text.upper())


def _article_labels(text: str) -> list[str]:
    return [f"Điều {value}" for value in re.findall(r"(?i)\bĐiều\s+(\d+[A-Za-z]?)", text)]


def _explicit_status_relations(article: ArticleNode) -> set[tuple[str, str]]:
    """Extract only sentence-local, explicit replacement/expiration statements."""
    output: set[tuple[str, str]] = set()
    for sentence in re.split(r"(?<=[.;])\s+|\n+", article.text):
        lowered = " ".join(sentence.lower().split())
        if not lowered:
            continue
        if any(marker in lowered for marker in ("hết hiệu lực", "chấm dứt hiệu lực", "bãi bỏ")):
            relation_type = "abolishes_doc_ids"
        elif any(marker in lowered for marker in ("thay thế", "thay cho")):
            relation_type = "replaces_doc_ids"
        else:
            continue
        for target_doc_id in _doc_ids(sentence):
            if target_doc_id.upper() != article.doc_id.upper():
                output.add((relation_type, target_doc_id))
    return output
