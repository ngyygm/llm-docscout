"""Generate synth-v2 SFT demos with VARIABLE read counts: read candidates in BM25
order until the gold section is read (capped), then answer. This yields 1-to-K read
trajectories where the count correlates with gold rank — giving the policy NATURAL
length variance so RL can later optimize WHEN to stop (the efficiency lever).

This is the SFT base for the RL>SFT accuracy-per-read-token experiment: RL's ratio
reward should learn to stop EARLIER on easy instances (gold at low idx) without
hurting accuracy → Pareto-dominate the fixed-length SFT.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from docscout.agent.rollout import SYSTEM_PROMPT
from docscout.data.synth_generator import generate_corpus
from docscout.data.sft_trajectories import _action_text, _user_turn
from docscout.env.docstore import DocStore
from docscout.env.search_env import SearchEnv, EnvConfig
from docscout.types import Action, ActionType


def build_rug_traj(inst, cap=4):
    """Read candidates in BM25 order until ALL gold sections read (or cap)."""
    store = DocStore(inst.docs)
    env = SearchEnv(inst, store, EnvConfig(max_steps=12))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    plan = [Action(ActionType.SEARCH, {"query": inst.question})]
    hits = store.search(inst.question, k=10)
    rank_uid = {(h["doc_id"], h["section_id"]): i + 1 for i, h in enumerate(hits)}
    gold_set = {(e.doc_id, e.section_id) for e in inst.gold_evidence}
    found_gold = set()
    used_idx = set()
    for h in hits[:cap]:
        uid = (h["doc_id"], h["section_id"])
        idx = rank_uid[uid]
        plan.append(Action(ActionType.READ, {"idx": idx}))
        used_idx.add(idx)
        if uid in gold_set:
            found_gold.add(uid)
        if len(found_gold) == len(gold_set):
            break  # found all gold — stop reading
    plan.append(Action(ActionType.ANSWER,
                       {"text": inst.gold_answer,
                        "evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in inst.gold_evidence]}))
    for a in plan:
        messages.append({"role": "user", "content": _user_turn(env)})
        step = env.step(a)
        messages.append({"role": "assistant", "content": _action_text(a)})
        messages.append({"role": "tool", "content": step.observation})
    return messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=800)
    p.add_argument("--out", default="data/sft/synth_rug_800.jsonl")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--cap", type=int, default=4)
    args = p.parse_args()
    random.seed(args.seed)
    insts = generate_corpus(n_docs=16, sections_per_doc=10, n_instances=args.n,
                            multi_hop_frac=0.2, paraphrase_frac=0.6, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    read_counts = []
    k = 0
    with open(args.out, "w") as f:
        for ins in insts:
            msgs = build_rug_traj(ins, args.cap)
            nread = sum(1 for m in msgs if m['role'] == 'assistant' and 'ACTION: read' in m['content'])
            read_counts.append(nread)
            f.write(json.dumps({"instance_id": ins.instance_id, "messages": msgs}, ensure_ascii=False) + "\n")
            k += 1
    import statistics as st
    print(f"wrote {k} read-until-gold demos -> {args.out}")
    print(f"read-count distribution: mean={st.mean(read_counts):.2f} "
          f"min={min(read_counts)} max={max(read_counts)} "
          f"(1:{read_counts.count(1)} 2:{read_counts.count(2)} 3:{read_counts.count(3)} 4:{read_counts.count(4)})")


if __name__ == "__main__":
    main()
