from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from legal_rag.schemas.models import ArticleNode
from legal_rag.utils.text import tokenize


class BM25Index:
    def __init__(self, articles: list[ArticleNode], k1: float = 1.5, b: float = 0.75) -> None:
        self.articles = articles
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(_article_search_text(article)) for article in articles]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freqs.update(set(tokens))

    def search(self, query: str, top_k: int = 20) -> list[ArticleNode]:
        query_terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, freqs in enumerate(self.term_freqs):
            score = self._score(query_terms, freqs, self.doc_lens[index])
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[ArticleNode] = []
        for score, index in scored[:top_k]:
            article = ArticleNode.from_dict(self.articles[index].to_dict())
            article.score = score
            results.append(article)
        return results

    def _score(self, query_terms: list[str], freqs: Counter[str], doc_len: int) -> float:
        score = 0.0
        total_docs = len(self.articles)
        if not total_docs or not self.avgdl:
            return 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"articles": [a.to_dict() for a in self.articles]}
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        articles = [ArticleNode.from_dict(item) for item in payload.get("articles", [])]
        return cls(articles)


def _article_search_text(article: ArticleNode) -> str:
    return " ".join(
        [
            article.doc_id,
            article.doc_type,
            article.title_for_submission,
            article.article_label,
            article.article_title,
            article.text,
        ]
    )

