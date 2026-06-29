"""Rule-based demonstration trajectories for SFT (自动化实验迭代方案.md §第三轮).

Before RL, the user's plan calls for SFT on demonstration trajectories. We
construct *ideal* trajectories rule-based from gold evidence (no model needed):
  search(question) -> read(each gold section) -> answer(gold_answer, gold_evidence)
These teach the basic action format + evidence-reading behavior; RL then optimizes
efficiency/stop policy on top. Output is chat-formatted JSONL ready for SFT.

Multi-section evidence may include an `expand` step to demonstrate the dynamic
reading window (gold in a neighbor).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docscout.agent.rollout import SYSTEM_PROMPT, build_messages  # reuse prompt
from docscout.env.docstore import DocStore
from docscout.env.search_env import SearchEnv, EnvConfig
from docscout.types import Action, ActionType, QAInstance


def _action_text(a: Action) -> str:
    """Render an Action as the exact ACTION-block text the parser expects."""
    if a.type == ActionType.SEARCH:
        return f"ACTION: search\nquery: {a.args.get('query','')}"
    if a.type == ActionType.READ:
        if "idx" in a.args:
            return f"ACTION: read\nidx: {a.args['idx']}"
        return f"ACTION: read\ndoc_id: {a.args['doc_id']}\nsection_id: {a.args['section_id']}"
    if a.type == ActionType.EXPAND:
        return f"ACTION: expand\ndoc_id: {a.args['doc_id']}\nsection_id: {a.args['section_id']}\ndirection: {a.args['direction']}"
    ev = ", ".join(f"{e['doc_id']}:{e['section_id']}" for e in a.args.get("evidence", []))
    return f"ACTION: answer\ntext: {a.args.get('text','')}\nevidence: {ev}"


def build_demo_trajectory(inst: QAInstance, max_reads: int = 3, adaptive: bool = False, topn: bool = False) -> list[dict]:
    """Return a chat message list (system, user, assistant, tool, ...) for SFT.

    topn=True (Round 11, the key fix): read the top-N candidates in rank order
    (N = number of gold sections), WITHOUT knowing which candidate is gold.
    This is the honest, eval-replicable policy. Crucially it is SUB-OPTIMAL vs
    the reward (when gold is ranked >N it is missed), so SFT is no longer at the
    reward maximum -> the RL reward is NON-FLAT at the SFT optimum -> RL finally
    has a gradient. The 'gold' and 'adaptive' modes cheat (they read gold by its
    known rank), which made SFT reward-optimal and RL inert (0/150 changed).
    """
    store = DocStore(inst.docs)
    env = SearchEnv(inst, store, EnvConfig(max_steps=10))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    plan: list[Action] = [Action(ActionType.SEARCH, {"query": inst.question})]
    hits = store.search(inst.question, k=10)
    rank_uid = {(h["doc_id"], h["section_id"]): i + 1 for i, h in enumerate(hits)}
    gold_sorted = sorted(inst.gold_evidence, key=lambda e: rank_uid.get((e.doc_id, e.section_id), 999))
    import random as _rng
    if topn:
        n = max(1, len(inst.gold_evidence))  # read top-N (1 for single-hop, 2 for multi-hop)
        for idx in range(1, min(n, 5) + 1):
            plan.append(Action(ActionType.READ, {"idx": idx}))
    elif adaptive:
        # read candidates 1..k in order, where k = rank of the deepest gold section
        gold_ranks = sorted({rank_uid.get((e.doc_id, e.section_id), 999) for e in inst.gold_evidence
                             if rank_uid.get((e.doc_id, e.section_id), 999) <= 10})
        max_needed = min(gold_ranks[-1] if gold_ranks else 1, 5)
        for idx in range(1, max_needed + 1):
            plan.append(Action(ActionType.READ, {"idx": idx}))
    else:
        # ADAPTIVE READ: if the gold answer value appears in the top-1 snippet, teach
        # search→answer (skip read, answer from snippet = CHEAP). Otherwise teach
        # search→read(gold)→answer (EXPENSIVE). This 50/50 mix creates the read/skip
        # diversity GRPO needs to learn the efficiency frontier.
        top1_snippet = hits[0]["snippet"] if hits else ""
        can_skip = str(inst.gold_answer).strip().lower() in top1_snippet.lower()
        if can_skip and _rng.random() < 0.5:
            pass  # skip read — answer directly from snippet (0 committed reads)
        else:
            used_idx = set()
            for e in gold_sorted[:max_reads]:
                idx = rank_uid.get((e.doc_id, e.section_id))
                if idx and idx not in used_idx:
                    plan.append(Action(ActionType.READ, {"idx": idx}))
                    used_idx.add(idx)
    plan.append(Action(ActionType.ANSWER,
                       {"text": inst.gold_answer,
                        "evidence": [{"doc_id": e.doc_id, "section_id": e.section_id} for e in gold_sorted]}))

    for a in plan:
        messages.append({"role": "user", "content": _user_turn(env)})
        step = env.step(a)
        messages.append({"role": "assistant", "content": _action_text(a)})
        messages.append({"role": "tool", "content": step.observation})
    return messages


def _user_turn(env: SearchEnv) -> str:
    """Reproduce the compact-memory user prompt (without system header)."""
    msgs = build_messages(env, recent_reads_kept=2)
    return msgs[-1]["content"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/grounded/musique_dev.json")
    p.add_argument("--out", default="data/sft/musique_demo.jsonl")
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--adaptive", action="store_true", help="rank-order reading until gold covered (Round 10)")
    p.add_argument("--topn", action="store_true", help="read top-N candidates non-cheating (Round 11, key fix)")
    args = p.parse_args()
    from scripts.run_retriever_eval import load_split
    insts = load_split(args.split)[: args.n]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    k = 0
    with open(args.out, "w") as f:
        for ins in insts:
            msgs = build_demo_trajectory(ins, adaptive=args.adaptive, topn=args.topn)
            f.write(json.dumps({"instance_id": ins.instance_id, "messages": msgs}, ensure_ascii=False) + "\n")
            k += 1
    print(f"wrote {k} demo trajectories -> {args.out}")


if __name__ == "__main__":
    main()
