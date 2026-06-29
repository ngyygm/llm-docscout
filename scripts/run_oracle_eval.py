"""Phase-1 Experiment 1 (自动化实验迭代方案.md): Oracle Evidence — solvability ceiling.

Give the model ONLY the gold evidence sections as context and ask it to answer.
If accuracy is high -> the task is solvable; the bottleneck is retrieval/selection
(not reading/answering) -> safe to train policy. If low -> fix data/task first.

Phase-2 baselines are also here (--mode {oneshot_rag, fixed_flow, prompted, oracle}):
  oracle      : gold-evidence context, answer            (Phase-1 Exp1 / ceiling)
  oneshot_rag : search top-k snippets as context, answer (most important control)
  fixed_flow  : heuristic search->read-top3->answer      (is agent deliberation needed?)
  prompted    : full agent loop, NO training             (what can the base model already do?)

Needs a served model: python -m scripts.run_oracle_eval --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import statistics as st

from docscout.agent.client import OpenAIClient
from docscout.agent.parsing import parse_action
from docscout.env.docstore import DocStore
from docscout.env.search_env import EnvConfig, SearchEnv
from docscout.agent.rollout import rollout
from docscout.reward.answer_scoring import normalize, score_answer
from docscout.types import Action, ActionType, QAInstance


def _load(path: str) -> list[QAInstance]:
    from scripts.run_retriever_eval import load_split
    return load_split(path)


def _em(pred: str, gold: str, aliases: list[str]) -> float:
    best = score_answer(pred, gold)
    for a in aliases or []:
        best = max(best, score_answer(pred, a))
    return best


def _gold_context(inst: QAInstance) -> str:
    parts = []
    for e in inst.gold_evidence:
        for d in inst.docs:
            if d.doc_id == e.doc_id:
                for s in d.sections:
                    if s.section_id == e.section_id:
                        parts.append(f"[{d.title}] {s.text}")
    return "\n".join(parts)


def oracle_or_rag_answer(client, inst: QAInstance, mode: str, k: int = 5) -> str:
    if mode == "oracle":
        ctx = _gold_context(inst)
    else:  # oneshot_rag
        store = DocStore(inst.docs)
        hits = store.search(inst.question, k=k)
        ctx = "\n".join(f"[{h['doc_title']}] {h['snippet']}" for h in hits)
    prompt = (f"Answer the question using ONLY the context below. Reply with just the answer.\n\n"
              f"Context:\n{ctx}\n\nQuestion: {inst.question}\nAnswer:")
    return client.complete(prompt, max_tokens=64, temperature=0.0)


def evaluate(client, instances, mode, n, k, shard=0, num_shards=1):
    sel = instances[:n][shard::num_shards] if num_shards > 1 else instances[:n]
    scores, samples = [], []
    for j, ins in enumerate(sel):
        if mode in ("oracle", "oneshot_rag"):
            pred = oracle_or_rag_answer(client, ins, mode, k)
            s = _em(pred, ins.gold_answer, ins.meta.get("answer_aliases", []))
        else:  # prompted / fixed_flow via full agent loop (base model, no training)
            client.temperature = 0.0
            res = rollout(ins, client, env_config=EnvConfig(max_steps=6, search_k=k), reward_name="ratio")
            pred = res.trajectory.final_answer
            s = _em(pred, ins.gold_answer, ins.meta.get("answer_aliases", []))
        scores.append(s)
        if len(samples) < 8 and shard == 0:
            samples.append({"q": ins.question[:80], "gold": ins.gold_answer, "pred": pred[:80], "score": s})
        if (j + 1) % 10 == 0:
            print(f"  [{mode} shard {shard}] {j+1}/{len(sel)} done", flush=True)
    return (st.mean(scores) if scores else 0.0), len(scores), samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/grounded/musique_dev.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--mode", default="oracle", choices=["oracle", "oneshot_rag", "prompted"])
    p.add_argument("--backend", default="hf", choices=["hf", "openai"])
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    insts = _load(args.split)
    if args.backend == "hf":
        from docscout.agent.client import HFClient
        client = HFClient(args.model)
    else:
        client = OpenAIClient(model=args.model, base_url=args.base_url)
    acc, n, samples = evaluate(client, insts, args.mode, args.n, args.k, args.shard, args.num_shards)
    print(f"=== {args.mode} (shard {args.shard}) on {n} instances ({args.model}) ===")
    print(f"  answer_acc (EM+partial) = {acc:.3f}")
    if args.out:
        json.dump({"mode": args.mode, "model": args.model, "n": n, "answer_acc": acc}, open(args.out, "w"), indent=2)
        if samples:
            json.dump(samples, open(args.out.replace(".json", "_samples.json"), "w"), indent=2, ensure_ascii=False)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
