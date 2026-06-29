"""Generate synth-v2 SFT demos that ALWAYS read top-3 candidates then answer.

Purpose: create a HIGH-COST SFT baseline (reads 3 every time) so RL has a clear
efficiency lever to optimize (learn to stop reading early when gold is found at
idx 1). This is the setup for the RL>SFT accuracy-per-read-token result.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from docscout.agent.rollout import SYSTEM_PROMPT, build_messages
from docscout.data.synth_generator import generate_corpus
from docscout.data.sft_trajectories import _action_text, _user_turn
from docscout.env.docstore import DocStore
from docscout.env.search_env import SearchEnv, EnvConfig
from docscout.types import Action, ActionType


def build_top3_traj(inst, max_reads=3):
    store = DocStore(inst.docs)
    env = SearchEnv(inst, store, EnvConfig(max_steps=10))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    plan = [Action(ActionType.SEARCH, {"query": inst.question})]
    hits = store.search(inst.question, k=10)
    rank_uid = {(h["doc_id"], h["section_id"]): i + 1 for i, h in enumerate(hits)}
    # ALWAYS read the top-3 candidates (high cost, but maximizes gold coverage)
    used_idx = set()
    for h in hits[:max_reads]:
        idx = rank_uid.get((h["doc_id"], h["section_id"]))
        if idx and idx not in used_idx:
            plan.append(Action(ActionType.READ, {"idx": idx}))
            used_idx.add(idx)
    gold_sorted = sorted(inst.gold_evidence, key=lambda e: rank_uid.get((e.doc_id, e.section_id), 999))
    plan.append(Action(ActionType.ANSWER,
                       {"text": inst.gold_answer,
                        "evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in gold_sorted]}))
    for a in plan:
        messages.append({"role": "user", "content": _user_turn(env)})
        step = env.step(a)
        messages.append({"role": "assistant", "content": _action_text(a)})
        messages.append({"role": "tool", "content": step.observation})
    return messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=800)
    p.add_argument("--out", default="data/sft/synth_top3_800.jsonl")
    p.add_argument("--seed", type=int, default=21)
    args = p.parse_args()
    random.seed(args.seed)
    insts = generate_corpus(n_docs=16, sections_per_doc=10, n_instances=args.n,
                            multi_hop_frac=0.2, paraphrase_frac=0.6, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    k = 0
    with open(args.out, "w") as f:
        for ins in insts:
            msgs = build_top3_traj(ins)
            f.write(json.dumps({"instance_id": ins.instance_id, "messages": msgs}, ensure_ascii=False) + "\n")
            k += 1
    print(f"wrote {k} top3 demo trajectories -> {args.out}")


if __name__ == "__main__":
    main()
