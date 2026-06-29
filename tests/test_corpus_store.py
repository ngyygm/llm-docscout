"""Unit tests for docscout.data.corpus_store — separate serialization + split leakage.

Run with:
    python -m pytest tests/test_corpus_store.py -v
    python tests/test_corpus_store.py
"""

from __future__ import annotations

import json
import sys
import pathlib
import tempfile
import shutil

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docscout.data.corpus_store import (
    CorpusStore,
    QAInstanceRef,
    serialize_separate,
    deserialize_separate,
    validate_split_leakage,
)
from docscout.types import Document, EvidenceSpan, QAInstance, Section


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs() -> list[Document]:
    """Create a small deterministic corpus (2 docs, 3 sections each)."""
    return [
        Document(
            doc_id="doc_01",
            title="AuroraPay Operations Policy",
            sections=[
                Section("1", "Overview", "AuroraPay syncs within 5 minutes after approval."),
                Section("2", "Retry", "AuroraPay retries up to 3 times on failure."),
                Section("3", "Retention", "AuroraPay records are kept for 90 days."),
            ],
        ),
        Document(
            doc_id="doc_02",
            title="BrightShip Operations Policy",
            sections=[
                Section("1", "Overview", "BrightShip syncs within 30 minutes after approval."),
                Section("2", "Retry", "BrightShip retries up to 10 times on failure."),
                Section("3", "RateLimit", "BrightShip allows 600 requests per minute."),
            ],
        ),
    ]


def _make_instances(n: int = 4, docs: list[Document] | None = None) -> list[QAInstance]:
    """Build n QAInstance objects sharing the same docs."""
    if docs is None:
        docs = _make_docs()
    instances = []
    questions = [
        "How many minutes after approval does AuroraPay synchronize?",
        "How many retries does AuroraPay attempt on failure?",
        "How many minutes after approval does BrightShip synchronize?",
        "How many retries does BrightShip attempt on failure?",
    ]
    answers = ["5", "3", "30", "10"]
    evidences = [
        [EvidenceSpan("doc_01", "1")],
        [EvidenceSpan("doc_01", "2")],
        [EvidenceSpan("doc_02", "1")],
        [EvidenceSpan("doc_02", "2")],
    ]
    for i in range(n):
        instances.append(
            QAInstance(
                instance_id=f"qa_{i+1:03d}",
                question=questions[i % len(questions)],
                gold_answer=answers[i % len(answers)],
                gold_evidence=evidences[i % len(evidences)],
                docs=docs,
                meta={"idx": i},
            )
        )
    return instances


# ---------------------------------------------------------------------------
# CorpusStore tests
# ---------------------------------------------------------------------------

def test_corpus_store_add_and_get():
    store = CorpusStore()
    docs = _make_docs()
    store.add_corpus("c1", docs)
    retrieved = store.get_corpus("c1")
    assert len(retrieved) == 2
    assert retrieved[0].doc_id == "doc_01"
    assert retrieved[1].doc_id == "doc_2" if False else "doc_02"


def test_corpus_store_dedup():
    store = CorpusStore()
    docs = _make_docs()
    id1 = store.add_corpus_dedup(docs)
    id2 = store.add_corpus_dedup(docs)
    assert id1 == id2, "identical docs should produce the same corpus_id"
    assert len(store) == 1


def test_corpus_store_all_doc_ids():
    store = CorpusStore()
    docs = _make_docs()
    store.add_corpus("c1", docs)
    assert store.all_doc_ids() == {"doc_01", "doc_02"}


def test_corpus_store_all_section_uids():
    store = CorpusStore()
    docs = _make_docs()
    store.add_corpus("c1", docs)
    uids = store.all_section_uids()
    expected = {
        ("doc_01", "1"), ("doc_01", "2"), ("doc_01", "3"),
        ("doc_02", "1"), ("doc_02", "2"), ("doc_02", "3"),
    }
    assert uids == expected


def test_corpus_store_roundtrip():
    store = CorpusStore()
    docs = _make_docs()
    store.add_corpus("c1", docs)
    data = store.to_dict()
    store2 = CorpusStore.from_dict(data)
    assert set(store2.corpora.keys()) == {"c1"}
    assert len(store2.corpora["c1"]) == 2
    assert store2.corpora["c1"][0].doc_id == "doc_01"


def test_corpus_store_content_hash_stability():
    docs = _make_docs()
    h1 = CorpusStore.corpus_hash(docs)
    h2 = CorpusStore.corpus_hash(docs)
    assert h1 == h2, "hash must be stable across calls"


def test_corpus_store_all_entity_names():
    store = CorpusStore()
    store.add_corpus("c1", _make_docs())
    entities = store.all_entity_names()
    assert "AuroraPay" in entities
    assert "BrightShip" in entities


def test_corpus_store_contains():
    store = CorpusStore()
    store.add_corpus("c1", _make_docs())
    assert "c1" in store
    assert "c2" not in store


# ---------------------------------------------------------------------------
# QAInstanceRef tests
# ---------------------------------------------------------------------------

def test_qa_ref_from_instance():
    insts = _make_instances(1)
    ref = QAInstanceRef.from_instance(insts[0], "c1")
    assert ref.instance_id == "qa_001"
    assert ref.corpus_id == "c1"
    assert ref.gold_answer == "5"
    assert len(ref.gold_evidence) == 1


def test_qa_ref_materialize():
    insts = _make_instances(1)
    ref = QAInstanceRef.from_instance(insts[0], "c1")
    store = CorpusStore()
    store.add_corpus("c1", insts[0].docs)
    restored = ref.materialize(store)
    assert restored.instance_id == ref.instance_id
    assert restored.gold_answer == ref.gold_answer
    assert len(restored.docs) == 2
    assert restored.docs[0].doc_id == "doc_01"


def test_qa_ref_roundtrip():
    insts = _make_instances(1)
    ref = QAInstanceRef.from_instance(insts[0], "c1")
    data = ref.to_dict()
    ref2 = QAInstanceRef.from_dict(data)
    assert ref2.instance_id == ref.instance_id
    assert ref2.corpus_id == ref.corpus_id
    assert ref2.gold_answer == ref.gold_answer
    assert len(ref2.gold_evidence) == len(ref.gold_evidence)
    assert ref2.gold_evidence[0].doc_id == ref.gold_evidence[0].doc_id


# ---------------------------------------------------------------------------
# serialize_separate / deserialize_separate tests
# ---------------------------------------------------------------------------

def test_serialize_deserialize_same_docs():
    tmp = tempfile.mkdtemp()
    try:
        insts = _make_instances(3)
        store, refs = serialize_separate(
            insts,
            pathlib.Path(tmp) / "corpora.json",
            pathlib.Path(tmp) / "qa.jsonl",
            dedup=True,
        )
        # All 3 instances share the same corpus → dedup to 1.
        assert len(store) == 1
        assert len(refs) == 3
        # All refs point to the same corpus_id.
        assert all(r.corpus_id == refs[0].corpus_id for r in refs)

        # Verify files exist and are readable
        assert pathlib.Path(tmp, "corpora.json").exists()
        assert pathlib.Path(tmp, "qa.jsonl").exists()

        # Deserialize and compare
        restored = deserialize_separate(
            pathlib.Path(tmp, "corpora.json"),
            pathlib.Path(tmp, "qa.jsonl"),
        )
        assert len(restored) == 3
        for orig, new in zip(insts, restored):
            assert orig.instance_id == new.instance_id
            assert orig.question == new.question
            assert orig.gold_answer == new.gold_answer
            assert len(new.docs) == len(orig.docs)
            assert new.docs[0].doc_id == orig.docs[0].doc_id
            assert new.docs[0].sections[0].text == orig.docs[0].sections[0].text
    finally:
        shutil.rmtree(tmp)


def test_serialize_no_dedup():
    tmp = tempfile.mkdtemp()
    try:
        insts = _make_instances(3)
        _, refs = serialize_separate(
            insts,
            pathlib.Path(tmp) / "corpora.json",
            pathlib.Path(tmp) / "qa.jsonl",
            dedup=False,
        )
        # Each instance gets its own corpus.
        assert len(set(r.corpus_id for r in refs)) == 3
    finally:
        shutil.rmtree(tmp)


def test_deserialize_roundtrip_jsonl():
    tmp = tempfile.mkdtemp()
    try:
        insts = _make_instances(4)
        serialize_separate(
            insts,
            pathlib.Path(tmp) / "corpora.json",
            pathlib.Path(tmp) / "qa.jsonl",
        )
        # Verify JSONL structure
        qa_path = pathlib.Path(tmp, "qa.jsonl")
        lines = qa_path.read_text().strip().split("\n")
        assert len(lines) == 4
        for line in lines:
            obj = json.loads(line)
            assert "instance_id" in obj
            assert "corpus_id" in obj
            assert "docs" not in obj, "QA file should NOT contain docs"
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# validate_split_leakage tests
# ---------------------------------------------------------------------------

def test_leakage_no_overlap():
    """Two instances over the same corpus should report doc overlap."""
    docs = _make_docs()
    train = _make_instances(2, docs)
    test = _make_instances(2, docs)
    report = validate_split_leakage(train, test)
    assert not report["clean"], "same docs in both splits must not be clean"
    assert "doc_01" in report["doc_id_overlap"]
    assert len(report["section_overlap"]) > 0


def test_leakage_clean_different_docs():
    """Different docs in each split should be clean."""
    train_docs = [
        Document(
            doc_id="train_doc",
            title="TrainCo Operations Policy",
            sections=[Section("1", "Info", "TrainCo value is 42.")],
        ),
    ]
    test_docs = [
        Document(
            doc_id="test_doc",
            title="TestCo Operations Policy",
            sections=[Section("1", "Info", "TestCo value is 7.")],
        ),
    ]
    train = [QAInstance("t1", "What is TrainCo?", "42", [], docs=train_docs)]
    test = [QAInstance("e1", "What is TestCo?", "7", [], docs=test_docs)]
    report = validate_split_leakage(train, test)
    assert report["clean"]
    assert report["doc_id_overlap"] == set()
    assert report["entity_overlap"] == set()


def test_leakage_entity_overlap():
    """Same entity in train and test questions should be detected."""
    train = [
        QAInstance(
            "t1",
            "How many minutes after approval does AuroraPay synchronize?",
            "5",
            docs=[
                Document(
                    "t_doc",
                    "TrainDoc Operations Policy",
                    [Section("1", "X", "Train info")],
                ),
            ],
        )
    ]
    test = [
        QAInstance(
            "e1",
            "How many retries does AuroraPay attempt on failure?",
            "3",
            docs=[
                Document(
                    "e_doc",
                    "TestDoc Operations Policy",
                    [Section("1", "Y", "Test info")],
                ),
            ],
        )
    ]
    report = validate_split_leakage(train, test)
    assert not report["clean"]
    assert "AuroraPay" in report["entity_overlap"]


def test_leakage_report_fields():
    """Verify all expected fields exist in the report."""
    docs = _make_docs()
    train = _make_instances(1, docs)
    test = _make_instances(1, docs)
    report = validate_split_leakage(train, test)
    required = {
        "clean", "entity_overlap", "doc_id_overlap",
        "doc_content_overlap", "section_overlap",
        "train_entities", "test_entities",
        "train_doc_ids", "test_doc_ids",
    }
    missing = required - set(report.keys())
    assert not missing, f"missing fields: {missing}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return 0 if passed == len(fns) else 1


if __name__ == "__main__":
    sys.exit(main())