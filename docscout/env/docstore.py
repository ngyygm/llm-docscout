"""Document store with snippet-only search, section read, and neighbor expand.

The retrieval unit is a *section*. `search` returns **snippets only** (never full
text) to control context growth; `read` returns one full section; `expand`
returns an adjacent section — the dynamic reading window over stable units
(RESEARCH_BRIEF.md §三/§四). Search is self-contained BM25 (no external deps) so
the env runs on CPU for R0 smoke tests; a dense retriever can be swapped in via
`Retriever` protocol without touching the env.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from docscout.types import Document, Section
from docscout.env.tokenizer import count_tokens

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def snippet(text: str, max_words: int = 40) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " …"


@dataclass
class SectionRef:
    """A flattened, addressable section in the store."""

    doc_id: str
    section_id: str
    title: str
    text: str
    doc_title: str
    pos: int  # index within its document
    n_sections_in_doc: int

    @property
    def token_len(self) -> int:
        return count_tokens(self.text)

    @property
    def uid(self) -> tuple[str, str]:
        return (self.doc_id, self.section_id)


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[SectionRef, float]]: ...


class BM25Retriever:
    """Self-contained BM25 over section text (title folded in for matching)."""

    def __init__(self, refs: list[SectionRef], k1: float = 1.5, b: float = 0.75):
        self.refs = refs
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(f"{r.title} {r.text}") for r in refs]
        self.N = len(refs)
        self.avgdl = (sum(len(d) for d in self.docs_tokens) / self.N) if self.N else 0.0
        self.df: dict[str, int] = {}
        for toks in self.docs_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in self.df.items()
        }

    def _score(self, q_terms: list[str], idx: int) -> float:
        toks = self.docs_tokens[idx]
        dl = len(toks) or 1
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in q_terms:
            if q not in self.idf:
                continue
            f = tf.get(q, 0)
            idf = self.idf[q]
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        return s

    def search(self, query: str, k: int) -> list[tuple[SectionRef, float]]:
        q_terms = tokenize(query)
        if not q_terms or self.N == 0:
            return []
        scored = [(self._score(q_terms, i), i) for i in range(self.N)]
        scored.sort(reverse=True)
        return [(self.refs[i], sc) for sc, i in scored[:k]]


_CE_CACHE: dict = {}

def _get_ce(rerank_model: str):
    """Singleton cross-encoder (load once, reuse). Uses sentence_transformers
    CrossEncoder which handles device/materialization correctly."""
    if rerank_model not in _CE_CACHE:
        from sentence_transformers import CrossEncoder
        _CE_CACHE[rerank_model] = CrossEncoder(rerank_model, local_files_only=True)
    return _CE_CACHE[rerank_model]


def _ce_score(ce, pairs):
    return ce.predict(pairs).tolist()


class RerankRetriever:
    """BM25 top-N pool -> cross-encoder rerank -> top-k. Lifts R@5 when gold is
    retrieved but poorly ranked (the paraphrase case: R@10 high, R@5 low)."""

    def __init__(self, refs: list[SectionRef], pool: int = 20,
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.bm25 = BM25Retriever(refs)
        self.pool = pool
        self.ce = _get_ce(rerank_model)

    def search(self, query: str, k: int) -> list[tuple[SectionRef, float]]:
        cand = self.bm25.search(query, self.pool)
        if not cand:
            return []
        pairs = [(query, (r.title + ". " + r.text)[:512]) for r, _ in cand]
        scores = _ce_score(self.ce, pairs)
        order = sorted(range(len(cand)), key=lambda i: -float(scores[i]))
        return [(cand[i][0], float(scores[i])) for i in order[:k]]


class DocStore:
    """The search/read/expand backend over a corpus."""

    def __init__(self, docs: list[Document], retriever_cls: type = BM25Retriever,
                 snippet_words: int = 40, rerank: bool = False):
        self.docs = {d.doc_id: d for d in docs}
        self.snippet_words = snippet_words
        # flatten
        self.refs: list[SectionRef] = []
        self.ref_index: dict[tuple[str, str], SectionRef] = {}
        for d in docs:
            for i, s in enumerate(d.sections):
                ref = SectionRef(
                    doc_id=d.doc_id,
                    section_id=s.section_id,
                    title=s.title,
                    text=s.text,
                    doc_title=d.title,
                    pos=i,
                    n_sections_in_doc=len(d.sections),
                )
                self.refs.append(ref)
                self.ref_index[(d.doc_id, s.section_id)] = ref
        self.retriever = retriever_cls(self.refs)
        if rerank:
            self.retriever = RerankRetriever(self.refs)

    # -- search -------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k **snippets** (not full text). Each hit carries ids + score."""
        hits = self.retriever.search(query, k)
        out = []
        for ref, score in hits:
            out.append(
                {
                    "doc_id": ref.doc_id,
                    "section_id": ref.section_id,
                    "doc_title": ref.doc_title,
                    "section_title": ref.title,
                    "snippet": snippet(ref.text, self.snippet_words),
                    "score": round(float(score), 4),
                }
            )
        return out

    # -- read ---------------------------------------------------------------
    def read(self, doc_id: str, section_id: str) -> dict | None:
        ref = self.ref_index.get((doc_id, section_id))
        if ref is None:
            return None
        return {
            "doc_id": doc_id,
            "section_id": section_id,
            "doc_title": ref.doc_title,
            "section_title": ref.title,
            "content": ref.text,
            "token_len": ref.token_len,
            "has_prev": ref.pos > 0,
            "has_next": ref.pos < ref.n_sections_in_doc - 1,
        }

    # -- expand -------------------------------------------------------------
    def expand(self, doc_id: str, section_id: str, direction: str) -> dict | None:
        """Return the neighbor section (left/right). None if no neighbor / bad args."""
        if direction not in ("left", "right"):
            return None
        ref = self.ref_index.get((doc_id, section_id))
        if ref is None:
            return None
        delta = -1 if direction == "left" else 1
        npos = ref.pos + delta
        if npos < 0 or npos >= ref.n_sections_in_doc:
            return None
        doc = self.docs[doc_id]
        nbr = doc.sections[npos]
        return {
            "doc_id": doc_id,
            "section_id": nbr.section_id,
            "doc_title": ref.doc_title,
            "section_title": nbr.title,
            "content": nbr.text,
            "token_len": nbr.token_len,
            "direction": direction,
        }

    def all_section_uids(self) -> set[tuple[str, str]]:
        return set(self.ref_index.keys())
