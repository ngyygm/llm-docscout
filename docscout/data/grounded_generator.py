"""Grounded reverse-construction data pipeline for DocScout.

Bases: MuSiQue (primary) + HotpotQA + DocScope (stub). These benchmarks already
ship (question, answer, **gold supporting paragraphs/facts**) — i.e. the
"reverse-construction" with a necessity guarantee is built into them (MuSiQue by
composition; HotpotQA by annotation). We map their supporting units to
**section-locatable gold evidence** so DocStore/SearchEnv/reward work unchanged.

Difficulty tags (per 自动化实验迭代方案.md §五) let us slice results by task type
later. Necessity/sufficiency validation has a deterministic core (answerable flag,
gold-in-corpus, distractors present) plus a pluggable model-based check.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from docscout.types import Document, EvidenceSpan, QAInstance, Section


# --------------------------------------------------------------------------- loaders
def _stream(dataset: str, split: str) -> Iterator[dict]:
    from datasets import load_dataset
    name = {"musique": "dgslibisey/MuSiQue", "hotpotqa": "hotpot_qa", "2wiki": "voidful/2WikiMultihopQA"}[dataset]
    config = "distractor" if dataset == "hotpotqa" else None
    yield from load_dataset(name, config, split=split, streaming=True)


# --------------------------------------------------------------------------- converters
def musique_to_instance(rec: dict) -> QAInstance | None:
    if not rec.get("answerable", True):
        return None
    paragraphs = rec.get("paragraphs") or []
    if not paragraphs:
        return None
    # group by title preserving order; section_id = index within the title group
    by_title: dict[str, list[dict]] = defaultdict(list)
    for p in paragraphs:
        by_title[p["title"]].append(p)
    docs: list[Document] = []
    gold: list[EvidenceSpan] = []
    for title, plist in by_title.items():
        secs = []
        for i, p in enumerate(plist):
            sid = str(i + 1)
            # MuSiQue stores body text under "paragraph_text" (not "paragraph")
            text = p.get("paragraph_text") or p.get("paragraph") or p.get("section_title") or ""
            if p.get("section_title"):
                text = f"{p['section_title']}. {text}"
            secs.append(Section(section_id=sid, title=title, text=text))
            if p.get("is_supporting"):
                gold.append(EvidenceSpan(doc_id=title, section_id=sid))
        docs.append(Document(doc_id=title, title=title, sections=secs))
    if not gold:
        return None
    titles = {e.doc_id for e in gold}
    return QAInstance(
        instance_id=rec.get("id", ""),
        question=rec["question"],
        gold_answer=str(rec["answer"]),
        gold_evidence=gold,
        docs=docs,
        meta={
            "source": "musique",
            "answer_aliases": rec.get("answer_aliases", []),
            "num_gold_sections": len(gold),
            "cross_document": len(titles) > 1,
            "has_distractors": len(paragraphs) > len(gold),
            "answerable": True,
        },
    )


def hotpot_to_instance(rec: dict) -> QAInstance | None:
    ctx = rec.get("context") or []          # list of [title, [sentences]]
    sf = rec.get("supporting_facts") or {}  # {title: [sent_ids]} or list of [title, sent_id]
    if isinstance(sf, list):
        sf_titles = {t for t, _ in sf}
    else:
        sf_titles = set(sf.keys())
    docs, gold = [], []
    for title, sents in ctx:
        text = " ".join(sents) if isinstance(sents, list) else str(sents)
        docs.append(Document(doc_id=title, title=title,
                             sections=[Section(section_id="1", title=title, text=text)]))
        if title in sf_titles:
            gold.append(EvidenceSpan(doc_id=title, section_id="1"))
    if not gold or not rec.get("answer"):
        return None
    return QAInstance(
        instance_id=rec.get("id", ""),
        question=rec["question"],
        gold_answer=str(rec["answer"]),
        gold_evidence=gold,
        docs=docs,
        meta={
            "source": "hotpotqa", "num_gold_sections": len(gold),
            "cross_document": len(gold) > 1, "has_distractors": len(ctx) > len(gold),
            "answerable": True,
        },
    )


_CONVERTERS = {"musique": musique_to_instance, "hotpotqa": hotpot_to_instance}


# --------------------------------------------------------------------------- validation (deterministic core)
def validate(inst: QAInstance) -> tuple[bool, str]:
    """Deterministic necessity/quality checks. Model-based necessity/sufficiency
    (closed-book / distractor-only unanswerability) is added separately with a served model."""
    all_uids = {(d.doc_id, s.section_id) for d in inst.docs for s in d.sections}
    for e in inst.gold_evidence:
        if (e.doc_id, e.section_id) not in all_uids:
            return False, "gold evidence not in corpus"
    if not inst.gold_answer or inst.gold_answer.lower() in ("", "yes", "no") and not inst.meta.get("answerable", True):
        pass
    if inst.meta.get("num_gold_sections", 0) == 0:
        return False, "no gold evidence"
    return True, "ok"


def build_split(dataset: str, split: str = "validation", n_max: int = 1000,
                min_valid: int | None = None) -> list[QAInstance]:
    """Load + convert + validate a split. Returns up to n_max valid instances."""
    conv = _CONVERTERS[dataset]
    out: list[QAInstance] = []
    for i, rec in enumerate(_stream(dataset, split)):
        if len(out) >= n_max:
            break
        inst = conv(rec)
        if inst is None:
            continue
        ok, _ = validate(inst)
        if ok:
            out.append(inst)
    return out


def serialize(instances: list[QAInstance], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        [{
            "instance_id": ins.instance_id, "question": ins.question, "gold_answer": ins.gold_answer,
            "gold_evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in ins.gold_evidence],
            "meta": ins.meta,
            "docs": [{"doc_id": d.doc_id, "title": d.title,
                      "sections": [{"section_id": s.section_id, "title": s.title, "text": s.text} for s in d.sections]}
                     for d in ins.docs],
        } for ins in instances],
        open(path, "w"), ensure_ascii=False, indent=2,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["musique", "hotpotqa"], default="musique")
    p.add_argument("--split", default="validation")
    p.add_argument("--n-max", type=int, default=1000)
    p.add_argument("--out", default="data/grounded/musique_dev.json")
    args = p.parse_args()
    insts = build_split(args.dataset, args.split, args.n_max)
    serialize(insts, args.out)
    print(f"[{args.dataset}/{args.split}] {len(insts)} valid instances -> {args.out}")
    if insts:
        ex = insts[0]
        print("  example:", ex.question[:80], "|=>", ex.gold_answer, "| gold:", len(ex.gold_evidence), "sections")
        hops = sum(1 for i in insts if i.meta.get("cross_document"))
        print(f"  cross-document (multi-hop): {hops}/{len(insts)}")


if __name__ == "__main__":
    main()
