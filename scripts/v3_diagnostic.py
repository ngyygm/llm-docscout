"""Round-6 diagnostic: does synth-v3 (realistic-scale) desaturate RAG and give
the read-budget thesis leverage? Measures solvability ceiling (oracle) vs RAG
variants, reporting BOTH word-count and honest BPE-token context cost.

Modes (all single-shot prompt -> answer, reader = ckpts/docscout-sft for
comparability with prior toy-synth numbers):
  oracle       : gold full section(s) as context            (solvability ceiling)
  rag_full_k   : top-k FULL sections as context              (fair strong RAG)
  rag_snip_k   : top-k 40-word snippets as context           (snippet-only RAG)

Headroom = oracle > rag_full  =>  there is accuracy room a learned policy can
capture. Efficiency leverage = rag_full uses k full sections (expensive) while an
agent that reads only the gold section matches it cheaply.

Also runs an action-stability smoke: does the SFT agent emit parseable
search/read/answer actions on synth-v3 (schema transfer)?

  python -m scripts.v3_diagnostic --model ckpts/docscout-sft --split data/synth/v3_eval300.json -n 150
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from docscout.agent.client import HFClient
from docscout.agent.parsing import parse_action
from docscout.agent.rollout import rollout
from docscout.env.docstore import DocStore
from docscout.env.search_env import EnvConfig
from docscout.reward.answer_scoring import score_answer
from docscout.types import Document, QAInstance, Section


def load_split(path):
    raw = json.load(open(path))
    out = []
    for r in raw:
        docs = [Document(doc_id=d["doc_id"], title=d["title"],
                         sections=[Section(section_id=s["section_id"], title=s["title"], text=s["text"]) for s in d["sections"]])
                for d in r["docs"]]
        from docscout.types import EvidenceSpan
        ge = [EvidenceSpan(doc_id=e["doc_id"], section_id=e["section_id"]) for e in r["gold_evidence"]]
        out.append(QAInstance(instance_id=r["instance_id"], question=r["question"],
                              gold_answer=str(r["gold_answer"]), gold_evidence=ge, docs=docs, meta=r.get("meta", {})))
    return out


def _gold_sections_text(inst):
    parts = []
    for e in inst.gold_evidence:
        for d in inst.docs:
            if d.doc_id == e.doc_id:
                for s in d.sections:
                    if s.section_id == e.section_id:
                        parts.append(f"[{d.title} / {s.title}] {s.text}")
    return "\n\n".join(parts)


def _rag_full_text(inst, k):
    ds = DocStore(inst.docs)
    hits = ds.search(inst.question, k=k)
    return "\n\n".join(f"[{h['doc_title']} / {h['section_title']}] {ds.read(h['doc_id'], h['section_id'])['content']}" for h in hits), hits


def _rag_snip_text(inst, k):
    ds = DocStore(inst.docs)
    hits = ds.search(inst.question, k=k)
    return "\n\n".join(f"[{h['doc_title']}] {h['snippet']}" for h in hits), hits


def _score(pred, inst):
    best = score_answer(pred, inst.gold_answer)
    for a in inst.meta.get("answer_aliases", []):
        best = max(best, score_answer(pred, str(a)))
    return best


def run_mode(client, insts, mode, k, tok):
    scores, ctx_words, ctx_toks = [], [], []
    for j, ins in enumerate(insts):
        if mode == "oracle":
            ctx = _gold_sections_text(ins)
        elif mode == "rag_full":
            ctx, _ = _rag_full_text(ins, k)
        else:
            ctx, _ = _rag_snip_text(ins, k)
        prompt = (f"Answer the question using ONLY the context below. Reply with just the answer.\n\n"
                  f"Context:\n{ctx}\n\nQuestion: {ins.question}\nAnswer:")
        pred = client.complete(prompt, max_tokens=48, temperature=0.0)
        scores.append(_score(pred, ins))
        ctx_words.append(len(ctx.split()))
        ctx_toks.append(len(tok(ctx)["input_ids"]))
        if (j + 1) % 30 == 0:
            print(f"  [{mode} k={k}] {j+1}/{len(insts)} acc={st.mean(scores):.3f}", flush=True)
    return {
        "mode": mode, "k": k, "n": len(insts),
        "acc": round(st.mean(scores), 4),
        "ctx_words_mean": round(st.mean(ctx_words), 1),
        "ctx_bpe_tokens_mean": round(st.mean(ctx_toks), 1),
    }


def action_smoke(model_path, insts, n=12):
    """Does the SFT agent emit parseable search/read/answer on synth-v3?"""
    client = HFClient(model_path, temperature=0.3, max_new_tokens=96)
    parsed, answered, nread = 0, 0, []
    for ins in insts[:n]:
        res, env = rollout(ins, client, env_config=EnvConfig(max_steps=6, search_k=5), return_env=True)
        # count parseable actions + whether it answered
        if res.trajectory.terminated_by == "answer":
            answered += 1
        nread.append(env.n_read)
        parsed += 1
    return {"n": n, "answered": answered, "answer_rate": round(answered / n, 2),
            "mean_n_read": round(st.mean(nread), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ckpts/docscout-sft")
    p.add_argument("--split", default="data/synth/v3_eval300.json")
    p.add_argument("-n", type=int, default=150)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="results/v3_diagnostic.json")
    p.add_argument("--no-smoke", action="store_true")
    args = p.parse_args()

    insts = load_split(args.split)[: args.n]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    client = HFClient(args.model, device=args.device, temperature=0.0, max_new_tokens=48)

    results = []
    for mode, k in [("oracle", None), ("rag_full", 1), ("rag_full", 3),
                    ("rag_full", 5), ("rag_snip", 5), ("rag_snip", 10)]:
        print(f"\n=== {mode} k={k} ===", flush=True)
        results.append(run_mode(client, insts, mode, k, tok))

    smoke = None
    if not args.no_smoke:
        print("\n=== action smoke (SFT agent on synth-v3) ===", flush=True)
        smoke = action_smoke(args.model, insts, n=12)
        print(f"  {smoke}", flush=True)

    summary = {"model": args.model, "split": args.split, "n": args.n,
               "modes": results, "action_smoke": smoke}
    print("\n===== ROUND-6 DIAGNOSTIC SUMMARY =====")
    for r in results:
        print(f"  {r['mode']:<10} k={r['k']}  acc={r['acc']:.3f}  ctx_words={r['ctx_words_mean']:.0f}  ctx_bpe={r['ctx_bpe_tokens_mean']:.0f}")
    if smoke:
        print(f"  action_smoke: answer_rate={smoke['answer_rate']} mean_n_read={smoke['mean_n_read']}")
    oracle = next(r for r in results if r["mode"] == "oracle")
    rag5 = next(r for r in results if r["mode"] == "rag_full" and r["k"] == 5)
    print(f"\n  headroom oracle-rag_full5 = {oracle['acc']-rag5['acc']:+.3f}  "
          f"(want >0 for RL headroom; oracle in (0.45,0.92) for solvable-but-not-saturated)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
