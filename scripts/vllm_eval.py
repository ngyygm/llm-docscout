"""Fast concurrent eval via a vLLM OpenAI server (server-side batching).

Replaces the slow per-GPU HF eval. One vLLM server + ThreadPoolExecutor
concurrent requests => orders of magnitude faster. Supports oracle / oneshot_rag
(prompt->answer) modes. (Multi-turn prompted/RL use scripts/run_rollout-style
loops; wire those to vLLM via OpenAIClient separately.)

  # start server first: vllm serve <model> --enforce-eager --port 8000
  python -m scripts.vllm_eval --mode oracle --split data/synth/v2_dev400.json -n 400
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor

from docscout.agent.client import OpenAIClient
from docscout.env.docstore import DocStore
from docscout.reward.answer_scoring import score_answer
from docscout.types import QAInstance


def _load(path):
    from scripts.run_retriever_eval import load_split
    return load_split(path)


def _gold_context(inst):
    parts = []
    for e in inst.gold_evidence:
        for d in inst.docs:
            if d.doc_id == e.doc_id:
                for s in d.sections:
                    if s.section_id == e.section_id:
                        parts.append(f"[{d.title}] {s.text}")
    return "\n".join(parts)


def _prompt(inst, mode, k):
    if mode == "oracle":
        ctx = _gold_context(inst)
    else:
        hits = DocStore(inst.docs).search(inst.question, k=k)
        ctx = "\n".join(f"[{h['doc_title']}] {h['snippet']}" for h in hits)
    is_mc = inst.gold_answer.strip().strip("()").lower() in "abcd" and len(inst.gold_answer.strip("() ")) <= 1
    instr = ("Reply with ONLY the correct option letter, e.g. (a)." if is_mc
             else "Reply with just the answer.")
    return (f"Answer the question using ONLY the context below. {instr}\n\n"
            f"Context:\n{ctx}\n\nQuestion: {inst.question}\nAnswer:")


def _mc_score(pred: str, gold: str) -> float | None:
    """If gold is an MCQA letter, extract the option letter from pred and compare."""
    g = gold.strip().strip("()").lower()
    if len(g) == 1 and g in "abcd":
        import re
        m = re.search(r"\(?([a-dA-D])\)?", pred.strip()[:6])
        return 1.0 if (m and m.group(1).lower() == g) else 0.0
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="oracle", choices=["oracle", "oneshot_rag"])
    p.add_argument("--split", default="data/synth/v2_dev400.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None)
    p.add_argument("-n", type=int, default=400)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    insts = _load(args.split)[: args.n]
    client = OpenAIClient(model=args.model or args.base_url, base_url=args.base_url)
    # resolve actual served model id
    import urllib.request
    served = json.loads(urllib.request.urlopen(args.base_url + "/models", timeout=5).read())["data"][0]["id"]
    client.model = served

    def work(i):
        ins = insts[i]
        pred = client.complete(_prompt(ins, args.mode, args.k), max_tokens=48, temperature=0.0)
        mc = _mc_score(pred, ins.gold_answer)
        if mc is not None:
            s = mc
        else:
            s = score_answer(pred, ins.gold_answer)
            for a in ins.meta.get("answer_aliases", []):
                s = max(s, score_answer(pred, a))
        return i, pred, s

    scores = [0.0] * len(insts)
    preds = [None] * len(insts)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, pred, s in ex.map(work, range(len(insts))):
            scores[i] = s; preds[i] = pred

    acc = st.mean(scores)
    para = [scores[i] for i in range(len(insts)) if insts[i].meta.get("paraphrased")]
    dire = [scores[i] for i in range(len(insts)) if not insts[i].meta.get("paraphrased")]
    md = st.mean(dire) if dire else 0.0
    mp = st.mean(para) if para else 0.0
    print(f"=== {args.mode} (vLLM, n={len(insts)}) acc={acc:.3f} "
          f"[direct={md:.3f} n={len(dire)}, para={mp:.3f} n={len(para)}]")
    if args.out:
        json.dump({"mode": args.mode, "n": len(insts), "answer_acc": acc,
                   "acc_direct": md, "acc_paraphrased": mp},
                  open(args.out, "w"), indent=2)
    # save a few samples
    for i in range(4):
        print(f"  gold={insts[i].gold_answer!r} pred={preds[i][:50]!r} sc={scores[i]}")


if __name__ == "__main__":
    main()
