"""Chain-following demonstration trajectories for synth-v4 (Round 12).

Builds SFT demos that TEACH the multi-hop chain following an agent must learn:
  search("{e0} {a0}") -> read e0.a0 section -> value v0
  search("{L0} {v0}") -> find e1 -> read e1.a1 section -> value v1
  ... -> answer(vK)

The search queries are exactly what a chain-following agent would form after
reading each hop's value (the next attribute name is in the question, the value
just read is in the observation). This is the skill that beats one-shot RAG
(which cannot form these intermediate queries). Reuses sft_trajectories helpers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from docscout.agent.rollout import SYSTEM_PROMPT
from docscout.agent.parsing import parse_action
from docscout.data.synth_v4_generator import ATTRS, ATTR_NAMES
from docscout.data.sft_trajectories import _action_text, _user_turn
from docscout.env.docstore import DocStore
from docscout.env.search_env import EnvConfig, SearchEnv
from docscout.types import Action, ActionType, QAInstance


def _attr_sid(attr):
    return str(ATTR_NAMES.index(attr) + 1)


def _doc_id_for_entity(inst: QAInstance, entity: str) -> str:
    title = f"{entity} Operations Policy"
    for d in inst.docs:
        if d.title == title:
            return d.doc_id
    return ""


def _find_hit_idx(hits, doc_id, section_id):
    for h in hits:
        if h["doc_id"] == doc_id and h["section_id"] == section_id:
            return hits.index(h) + 1
    return None


def build_chain_demo(inst: QAInstance) -> list[dict]:
    store = DocStore(inst.docs)
    env = SearchEnv(inst, store, EnvConfig(max_steps=20))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    meta = inst.meta
    ents = meta["chain_ents"]; val_attrs = meta["chain_val_attrs"]
    link_attrs = meta["chain_link_attrs"]; values = meta["chain_values"]

    plan: list[Action] = []
    # hop 0: search e0's a0 section
    e0, a0 = ents[0], val_attrs[0]
    q0 = f"{e0} {ATTRS[a0][0]}"
    hits0 = store.search(q0, k=5)
    idx0 = _find_hit_idx(hits0, _doc_id_for_entity(inst, e0), _attr_sid(a0))
    if idx0:
        plan.append(Action(ActionType.SEARCH, {"query": q0}))
        plan.append(Action(ActionType.READ, {"idx": idx0}))
    # hops 1..K: search the link value, then read the next entity's value section
    for i in range(1, len(ents)):
        Lprev = link_attrs[i - 1]; vprev = values[i - 1]
        qL = f"{ATTRS[Lprev][0]} {vprev}"
        hits = store.search(qL, k=5)
        ei = ents[i]; ai = val_attrs[i]
        # the link hit reveals ei (its Lprev section); read ei's ai section
        idxL = _find_hit_idx(hits, _doc_id_for_entity(inst, ei), _attr_sid(Lprev))
        if idxL is None:
            break
        plan.append(Action(ActionType.SEARCH, {"query": qL}))
        plan.append(Action(ActionType.READ, {"doc_id": _doc_id_for_entity(inst, ei), "section_id": _attr_sid(ai)}))
    plan.append(Action(ActionType.ANSWER, {"text": inst.gold_answer, "evidence": []}))

    # render messages with env stepping
    for a in plan:
        messages.append({"role": "user", "content": _user_turn(env)})
        step = env.step(a)
        messages.append({"role": "assistant", "content": _action_text(a)})
        messages.append({"role": "tool", "content": step.observation})
    return messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/synth/v4_train1k.json")
    p.add_argument("--out", default="data/sft/v4_chain1k.jsonl")
    p.add_argument("-n", type=int, default=1000)
    args = p.parse_args()
    from scripts.run_retriever_eval import load_split
    insts = load_split(args.split)[: args.n]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    k = 0; skips = 0
    with open(args.out, "w") as f:
        for ins in insts:
            msgs = build_chain_demo(ins)
            if len(msgs) < 4:
                skips += 1; continue
            f.write(json.dumps({"instance_id": ins.instance_id, "messages": msgs}, ensure_ascii=False) + "\n")
            k += 1
    print(f"wrote {k} chain demos -> {args.out} (skipped {skips})")


if __name__ == "__main__":
    main()
