"""PersonaMem adapter — real long-context, multiple-choice, section-structured
read-budget eval for DocScout.

Maps each PersonaMem persona context (a multi-session user-model conversation) to:
  doc = the conversation; sections = token-bounded chunks of messages (~20 per inst).
  question = user_question_or_message + the answer options.
  gold_answer = correct_answer (an option phrase; EM-checkable, no LLM-judge).
  gold_evidence = the section(s) containing the correct_answer (the persona fact).

This gives a REAL read-budget task: the agent must find the right conversation
chunk among ~20 and pick the matching option. distance_to_ref_proportion_in_context
is preserved in meta as a difficulty knob.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docscout.types import Document, EvidenceSpan, QAInstance, Section


def _load_contexts(path: str) -> dict:
    ctx = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ctx.update(d)
    return ctx


def _chunk_sections(messages: list[dict], target_tokens: int = 1500) -> list[Section]:
    """Group messages into token-bounded sections (~target_tokens each)."""
    secs, buf, tok = [], [], 0
    for m in messages:
        c = m.get("content", "")
        t = len(c.split())
        buf.append(f"[{m.get('role','?')}] {c}")
        tok += t
        if tok >= target_tokens:
            secs.append(Section(section_id=str(len(secs) + 1),
                                title=f"Conversation chunk {len(secs)+1}",
                                text="\n".join(buf)))
            buf, tok = [], 0
    if buf:
        secs.append(Section(section_id=str(len(secs) + 1),
                            title=f"Conversation chunk {len(secs)+1}", text="\n".join(buf)))
    return secs


def _gold_sections(secs: list[Section], dprop: float) -> list[EvidenceSpan]:
    """Gold = the section at the proportional position of the preference mention
    (PersonaMem's distance_to_ref_proportion_in_context), + its neighbor for
    robustness. This aligns with the benchmark's evidence annotation."""
    n = len(secs)
    if n == 0:
        return []
    center = max(0, min(n - 1, int(round(dprop * (n - 1)))))
    idxs = sorted({max(0, center - 1), center, min(n - 1, center + 1)})
    return [EvidenceSpan("persona", secs[i].section_id) for i in idxs]


def build_instances(questions_csv: str, contexts_jsonl: str, n_max: int = 200) -> list[QAInstance]:
    ctx = _load_contexts(contexts_jsonl)
    out = []
    with open(questions_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if len(out) >= n_max:
                break
            cid = row["shared_context_id"]
            if cid not in ctx:
                continue
            end = int(float(row.get("end_index_in_shared_context") or len(ctx[cid])))
            messages = ctx[cid][:end]
            secs = _chunk_sections(messages)
            if not secs:
                continue
            opts = row["all_options"]
            dprop = row.get("distance_to_ref_proportion_in_context", "0")
            dprop = float(str(dprop).strip().rstrip("%")) / (100.0 if "%" in str(dprop) else 1.0)
            q = (f"{row['user_question_or_message']}\n\nOptions:\n{opts}")
            gold = _gold_sections(secs, dprop)
            doc = Document(doc_id="persona", title=f"Persona {row['persona_id']}", sections=secs)
            out.append(QAInstance(
                instance_id=row["question_id"], question=q, gold_answer=row["correct_answer"],
                gold_evidence=gold, docs=[doc],
                meta={"source": "personamem", "qtype": row.get("question_type", ""),
                      "distance_to_ref_proportion": dprop,
                      "num_gold_sections": len(gold), "options": opts, "answerable": True},
            ))
    return out


def serialize(instances, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump([{
        "instance_id": i.instance_id, "question": i.question, "gold_answer": i.gold_answer,
        "gold_evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in i.gold_evidence],
        "meta": i.meta,
        "docs": [{"doc_id": d.doc_id, "title": d.title,
                  "sections": [{"section_id": s.section_id, "title": s.title, "text": s.text} for s in d.sections]}
                 for d in i.docs],
    } for i in instances], open(path, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="data/personamem/questions_32k.csv")
    p.add_argument("--contexts", default="data/personamem/shared_contexts_32k.jsonl")
    p.add_argument("--n-max", type=int, default=200)
    p.add_argument("--out", default="data/personamem/dev200.json")
    args = p.parse_args()
    insts = build_instances(args.questions, args.contexts, args.n_max)
    serialize(insts, args.out)
    print(f"[personamem] {len(insts)} instances -> {args.out}")
    if insts:
        ex = insts[0]
        print(f"  example q: {ex.question[:80]}... | gold: {ex.gold_answer[:30]} | sections: {len(ex.docs[0].sections)} | gold_secs: {ex.meta['num_gold_sections']}")
