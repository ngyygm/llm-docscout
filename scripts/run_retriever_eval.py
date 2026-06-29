"""Phase-1 Experiment 3 (自动化实验迭代方案.md): Retriever Upper Bound.

Pure-retrieval diagnostic — NO agent, NO training. For each instance, query with
the raw question and measure whether BM25 surfaces a gold section in top-k.
Answers the doc's key gate: "is gold evidence retrievable at all?"
  Recall@5 high  -> search is fine; agent's job is selection + stopping.
  Recall@k low   -> fix retrieval (hybrid/reranker/chunking) BEFORE any RL.

Usage: python -m scripts.run_retriever_eval --split data/grounded/musique_dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict

from docscout.env.docstore import DocStore
from docscout.types import Document, EvidenceSpan, QAInstance, Section


def load_split(path: str) -> list[QAInstance]:
    raw = json.load(open(path))
    out = []
    for r in raw:
        docs = [Document(doc_id=d["doc_id"], title=d["title"],
                         sections=[Section(section_id=s["section_id"], title=s["title"], text=s["text"]) for s in d["sections"]])
                for d in r["docs"]]
        gold = [EvidenceSpan(doc_id=e["doc_id"], section_id=e["section_id"]) for e in r["gold_evidence"]]
        out.append(QAInstance(r["instance_id"], r["question"], r["gold_answer"], gold, docs, r.get("meta", {})))
    return out


def evaluate(instances, ks=(1, 5, 10)):
    per_k = {k: [] for k in ks}
    mrr, mrr_by = [], defaultdict(list)
    for ins in instances:
        store = DocStore(ins.docs)
        gold = ins.gold_sections()
        hits = store.search(ins.question, k=max(ks))
        hit_uids = [(h["doc_id"], h["section_id"]) for h in hits]
        ranks = [i + 1 for i, uid in enumerate(hit_uids) if uid in gold]
        for k in ks:
            per_k[k].append(1.0 if ranks and ranks[0] <= k else 0.0)
        rr = (1.0 / ranks[0]) if ranks else 0.0
        mrr.append(rr)
        mrr_by["cross" if ins.meta.get("cross_document") else "single"].append(rr)
    print(f"=== Retriever Upper Bound on {len(instances)} instances ===")
    for k in ks:
        print(f"  Recall@{k:<2} = {st.mean(per_k[k]):.3f}")
    print(f"  MRR       = {st.mean(mrr):.3f}")
    for g, vals in mrr_by.items():
        print(f"  MRR[{g}-doc] = {st.mean(vals):.3f}  (n={len(vals)})")
    return {f"recall@{k}": st.mean(per_k[k]) for k in ks} | {"mrr": st.mean(mrr), "n": len(instances)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/grounded/musique_dev.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    insts = load_split(args.split)
    res = evaluate(insts)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
