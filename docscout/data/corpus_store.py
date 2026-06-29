"""Corpus Store: manage independent corpus objects by corpus_id.

Provides separate serialization of corpus + QA instances so the corpus is stored
once and QA instances reference it via corpus_id — eliminating the duplication
where every QAInstance embeds a full copy of `docs`.

Motivation: DocScout synthetic generators produce many QA instances over the
same corpus. Embedding the full corpus in each QAInstance inflates JSON files
by N_instances * |corpus| and makes train/test split leakage detection
cumbersome.

Usage:
    from docscout.data.corpus_store import (
        CorpusStore, QAInstanceRef, serialize_separate, deserialize_separate,
        validate_split_leakage,
    )

    # Serialize: QAInstance -> (corpora.jsonl, qa.jsonl)
    serialize_separate(instances, "corpora.jsonl", "qa.jsonl")

    # Deserialize: reconstruct QAInstance list
    restored = deserialize_separate("corpora.jsonl", "qa.jsonl")

    # Leakage check
    report = validate_split_leakage(train_insts, test_insts)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docscout.types import Document, EvidenceSpan, QAInstance, Section


# ---------------------------------------------------------------------------
# CorpusStore
# ---------------------------------------------------------------------------

@dataclass
class CorpusStore:
    """Manages independent corpus objects indexed by corpus_id.

    Each corpus_id maps to a list[Document].  Multiple QA instances can share
    the same corpus (common in synthetic generation where the whole corpus is
    the same for every instance).  Deduplication is by corpus content hash,
    not by reference identity.
    """

    corpora: dict[str, list[Document]] = field(default_factory=dict)
    _doc_index: dict[str, Document] = field(default_factory=dict)

    def add_corpus(self, corpus_id: str, docs: list[Document]) -> None:
        """Register a corpus under a unique id."""
        if corpus_id in self.corpora:
            raise KeyError(f"corpus_id {corpus_id!r} already exists in store")
        self.corpora[corpus_id] = docs
        for d in docs:
            self._doc_index[d.doc_id] = d

    def get_corpus(self, corpus_id: str) -> list[Document]:
        """Return the documents for a corpus_id."""
        if corpus_id not in self.corpora:
            raise KeyError(f"corpus_id {corpus_id!r} not found")
        return self.corpora[corpus_id]

    def get_document(self, doc_id: str) -> Document | None:
        """Look up a single document by doc_id across all corpora."""
        return self._doc_index.get(doc_id)

    @staticmethod
    def corpus_hash(docs: list[Document]) -> str:
        """Content-based hash for deduplication.

        Hashes the concatenation of all section texts + titles, stable under
        Python's built-in hash randomization (uses deterministic SHA256).
        """
        import hashlib

        parts: list[str] = []
        for d in docs:
            parts.append(d.doc_id)
            parts.append(d.title)
            for s in d.sections:
                parts.append(s.section_id)
                parts.append(s.title)
                parts.append(s.text)
        payload = "\x00".join(parts).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def add_corpus_dedup(self, docs: list[Document]) -> str:
        """Add a corpus only if not already present; return its id."""
        cid = self.corpus_hash(docs)
        if cid not in self.corpora:
            self.add_corpus(cid, docs)
        return cid

    def all_doc_ids(self) -> set[str]:
        """Return the set of all doc_ids across all corpora."""
        return set(self._doc_index.keys())

    def all_section_uids(self) -> set[tuple[str, str]]:
        """Return all (doc_id, section_id) tuples across all corpora."""
        uids: set[tuple[str, str]] = set()
        for docs in self.corpora.values():
            for d in docs:
                for s in d.sections:
                    uids.add((d.doc_id, s.section_id))
        return uids

    def all_entity_names(self) -> set[str]:
        """Extract entity names from doc titles (heuristic: 'X Operations Policy' -> 'X')."""
        entities: set[str] = set()
        for docs in self.corpora.values():
            for d in docs:
                # Title is typically "AuroraPay Operations Policy"
                suffix = " Operations Policy"
                if d.title.endswith(suffix):
                    entities.add(d.title[: -len(suffix)])
        return entities

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe)."""
        out: dict[str, list[dict]] = {}
        for cid, docs in self.corpora.items():
            out[cid] = [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "sections": [
                        {"section_id": s.section_id, "title": s.title, "text": s.text}
                        for s in d.sections
                    ],
                }
                for d in docs
            ]
        return out

    @classmethod
    def from_dict(cls, data: dict) -> CorpusStore:
        """Deserialize from a plain dict."""
        store = cls()
        for cid, doc_list in data.items():
            docs = []
            for d in doc_list:
                sections = [
                    Section(
                        section_id=s["section_id"],
                        title=s["title"],
                        text=s["text"],
                    )
                    for s in d["sections"]
                ]
                docs.append(Document(doc_id=d["doc_id"], title=d["title"], sections=sections))
            store.add_corpus(cid, docs)
        return store

    def __len__(self) -> int:
        return len(self.corpora)

    def __contains__(self, corpus_id: str) -> bool:
        return corpus_id in self.corpora


# ---------------------------------------------------------------------------
# QAInstanceRef — lightweight reference to a corpus
# ---------------------------------------------------------------------------

@dataclass
class QAInstanceRef:
    """A QAInstance that references a corpus by corpus_id instead of embedding docs.

    This is the serializable form: corpus is stored once, many QAInstanceRefs
    point to it.  Use `materialize(store)` to reconstruct a full QAInstance.
    """

    instance_id: str
    question: str
    gold_answer: str
    corpus_id: str
    gold_evidence: list[EvidenceSpan] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def materialize(self, store: CorpusStore) -> QAInstance:
        """Reconstruct a full QAInstance by looking up the corpus in the store."""
        docs = store.get_corpus(self.corpus_id)
        return QAInstance(
            instance_id=self.instance_id,
            question=self.question,
            gold_answer=self.gold_answer,
            gold_evidence=list(self.gold_evidence),
            docs=docs,
            meta=dict(self.meta),
        )

    @staticmethod
    def from_instance(instance: QAInstance, corpus_id: str) -> QAInstanceRef:
        """Create a ref from a full QAInstance, stripping the docs."""
        return QAInstanceRef(
            instance_id=instance.instance_id,
            question=instance.question,
            gold_answer=instance.gold_answer,
            corpus_id=corpus_id,
            gold_evidence=list(instance.gold_evidence),
            meta=dict(instance.meta),
        )

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "corpus_id": self.corpus_id,
            "gold_evidence": [
                {"doc_id": e.doc_id, "section_id": e.section_id}
                for e in self.gold_evidence
            ],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> QAInstanceRef:
        return cls(
            instance_id=data["instance_id"],
            question=data["question"],
            gold_answer=data["gold_answer"],
            corpus_id=data["corpus_id"],
            gold_evidence=[
                EvidenceSpan(doc_id=e["doc_id"], section_id=e["section_id"])
                for e in data.get("gold_evidence", [])
            ],
            meta=data.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Separate serialization
# ---------------------------------------------------------------------------

def serialize_separate(
    qa_instances: list[QAInstance],
    corpora_path: str | Path,
    qa_path: str | Path,
    dedup: bool = True,
) -> tuple[CorpusStore, list[QAInstanceRef]]:
    """Split QAInstances into (corpora.jsonl, qa.jsonl).

    Args:
        qa_instances: Full QAInstance objects (with embedded docs).
        corpora_path: Where to write the corpus store (JSON).
        qa_path: Where to write the QA refs (JSONL — one object per line).
        dedup: If True, use content-hash dedup so identical corpora share one id.

    Returns:
        (store, refs) — the store and ref list that were written.
    """
    corpora_path = Path(corpora_path)
    qa_path = Path(qa_path)
    corpora_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)

    store = CorpusStore()
    refs: list[QAInstanceRef] = []

    for ins in qa_instances:
        if dedup:
            cid = store.add_corpus_dedup(ins.docs)
        else:
            cid = f"corpus_{ins.instance_id}"
            if cid not in store:
                store.add_corpus(cid, ins.docs)
        ref = QAInstanceRef.from_instance(ins, cid)
        refs.append(ref)

    # Write corpora as a single JSON object
    corpora_path.write_text(
        json.dumps(store.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Write QA as JSONL
    with open(qa_path, "w", encoding="utf-8") as f:
        for ref in refs:
            f.write(json.dumps(ref.to_dict(), ensure_ascii=False) + "\n")

    return store, refs


def deserialize_separate(
    corpora_path: str | Path,
    qa_path: str | Path,
) -> list[QAInstance]:
    """Reconstruct full QAInstances from separate corpus + QA files.

    Args:
        corpora_path: Path to the corpus store JSON (as written by serialize_separate).
        qa_path: Path to the QA JSONL file.

    Returns:
        list[QAInstance] with docs materialized from the store.
    """
    corpora_path = Path(corpora_path)
    qa_path = Path(qa_path)

    store_data = json.loads(corpora_path.read_text(encoding="utf-8"))
    store = CorpusStore.from_dict(store_data)

    instances: list[QAInstance] = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ref = QAInstanceRef.from_dict(json.loads(line))
            instances.append(ref.materialize(store))

    return instances


# ---------------------------------------------------------------------------
# Split leakage detection
# ---------------------------------------------------------------------------

def _extract_entities_from_question(question: str) -> set[str]:
    """Heuristic entity extraction: CamelCase tokens that look like entity names.

    Filters: at least 2 chars, starts with capital, and additional heuristics
    to avoid false positives on common first-words of questions (What, How, Which).
    """
    import re

    # Common question-starting words that should not count as entities
    _QUESTION_STOPS = {"What", "How", "Which", "Why", "When", "Where", "Who", "Is", "Are", "Do", "Does", "Can", "Could", "Would", "Will", "Should", "Has", "Have", "Did", "Was", "Were"}

    tokens = re.findall(r"[A-Z][a-zA-Z0-9]+", question)
    # Keep tokens that are not question-starters and have >= 3 characters
    # (longer tokens are more likely to be real entity names like AuroraPay)
    return {t for t in tokens if t not in _QUESTION_STOPS and len(t) >= 3}


def _extract_entities_from_doc_title(title: str) -> set[str]:
    """Extract entity name from doc title ('X Operations Policy' -> 'X')."""
    suffix = " Operations Policy"
    if title.endswith(suffix):
        return {title[: -len(suffix)]}
    return set()


def validate_split_leakage(
    train_instances: list[QAInstance],
    test_instances: list[QAInstance],
) -> dict:
    """Check for entity/document/doc_id leakage between train and test splits.

    Returns a dict with:
        - "clean": bool — True if no leakage found.
        - "entity_overlap": set[str] — entities that appear in both splits.
        - "doc_id_overlap": set[str] — doc_ids that appear in both splits.
        - "doc_content_overlap": set[str] — doc_ids whose content is identical
          (by section text hash) across splits.
        - "section_overlap": set[tuple[str, str]] — (doc_id, section_id) pairs
          that appear in both splits.
        - "train_entities": set[str]
        - "test_entities": set[str]
        - "train_doc_ids": set[str]
        - "test_doc_ids": set[str]
    """
    # Collect doc_ids and section uids per split
    def _collect(instances: list[QAInstance]):
        doc_ids: set[str] = set()
        sections: set[tuple[str, str]] = set()
        entities: set[str] = set()
        doc_content: dict[str, str] = {}  # doc_id -> content hash
        for ins in instances:
            for d in ins.docs:
                doc_ids.add(d.doc_id)
                entities |= _extract_entities_from_doc_title(d.title)
                # Content hash for content-level comparison
                parts = [d.doc_id, d.title]
                for s in d.sections:
                    parts.append(s.section_id)
                    parts.append(s.title)
                    parts.append(s.text)
                    sections.add((d.doc_id, s.section_id))
                import hashlib

                doc_content[d.doc_id] = hashlib.sha256(
                    "\x00".join(parts).encode("utf-8")
                ).hexdigest()[:16]
        # Also extract entities from questions
        for ins in instances:
            entities |= _extract_entities_from_question(ins.question)
        return doc_ids, sections, entities, doc_content

    train_doc_ids, train_sections, train_entities, train_content = _collect(train_instances)
    test_doc_ids, test_sections, test_entities, test_content = _collect(test_instances)

    entity_overlap = train_entities & test_entities
    doc_id_overlap = train_doc_ids & test_doc_ids
    section_overlap = train_sections & test_sections

    # Content-level overlap: same hash but possibly different doc_id
    train_hashes = {h: did for did, h in train_content.items()}
    test_hashes = {h: did for did, h in test_content.items()}
    common_hashes = set(train_hashes.keys()) & set(test_hashes.keys())
    doc_content_overlap: set[str] = set()
    for h in common_hashes:
        doc_content_overlap.add(train_hashes[h])
        doc_content_overlap.add(test_hashes[h])

    clean = not (entity_overlap or doc_id_overlap or doc_content_overlap or section_overlap)

    return {
        "clean": clean,
        "entity_overlap": entity_overlap,
        "doc_id_overlap": doc_id_overlap,
        "doc_content_overlap": doc_content_overlap,
        "section_overlap": section_overlap,
        "train_entities": train_entities,
        "test_entities": test_entities,
        "train_doc_ids": train_doc_ids,
        "test_doc_ids": test_doc_ids,
    }
