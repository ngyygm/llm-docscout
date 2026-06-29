"""Prompted-agent baseline + failure classification via vLLM (concurrent).

Phase-2 'prompted' baseline (base model, NO training, multi-step agent loop) on
the middle-difficulty substrate, plus the §四 failure-type breakdown. Uses vLLM
for fast batched rollouts (concurrent instances; vLLM batches across instances
and steps). Emits answer acc, read-token stats, and failure distribution.

  python -m scripts.vllm_prompted_eval --split data/synth/v2_dev400.json -n 200
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor

from docscout.agent.client import OpenAIClient
from docscout.agent.rollout import rollout
from docscout.env.search_env import EnvConfig
from docscout.eval.failure_analysis import analyze
from docscout.reward.answer_scoring import score_answer


def _load(path):
    from scripts.run_retriever_eval import load_split
    return load_split(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/synth/v2_dev400.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="results/v2_prompted.json")
    args = p.parse_args()

    insts = _load(args.split)[: args.n]
    client = OpenAIClient(model="x", base_url=args.base_url, temperature=0.0, max_tokens=96)
    import urllib.request
    client.model = json.loads(urllib.request.urlopen(args.base_url + "/models", timeout=5).read())["data"][0]["id"]

    records, scores, reads, steps = [], [], [], []

    def work(ins):
        res, env = rollout(ins, client, env_config=EnvConfig(max_steps=args.max_steps, search_k=args.k),
                           reward_name="ratio", return_env=True)
        s = score_answer(res.trajectory.final_answer, ins.gold_answer)
        for a in ins.meta.get("answer_aliases", []):
            s = max(s, score_answer(res.trajectory.final_answer, a))
        return ins, env, s, res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ins, env, s, res in ex.map(work, insts):
            records.append((env, ins)); scores.append(s)
            reads.append(res.components["read_tokens"]); steps.append(res.components["n_steps"])

    acc = st.mean(scores)
    fb = analyze(records)
    print(f"=== prompted (vLLM, n={len(insts)}) ===")
    print(f"  answer_acc = {acc:.3f}   mean_read_tokens = {st.mean(reads):.0f}   mean_steps = {st.mean(steps):.1f}")
    print("  failure breakdown:")
    for k in sorted(fb.distribution, key=lambda x: -fb.distribution[x]):
        print(f"    {k:14s} {fb.distribution[k]*100:5.1f}%  (n={fb.counts[k]})")
    if args.out:
        json.dump({"mode": "prompted", "n": len(insts), "answer_acc": acc,
                   "mean_read_tokens": st.mean(reads), "mean_steps": st.mean(steps),
                   "failure": fb.distribution}, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
