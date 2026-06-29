"""渐进式多跳SFT示范生成器。

核心思想：将chain-following训练分解为3个阶段，每个阶段只暴露
一种复杂度的demo，让模型逐步学会"extract value → form next query"模式。

阶段划分：
- Phase 1: 单步读取 (3 actions: search→read→answer)
  只练"读了就能答"，不练链式跟随
- Phase 2: 双步链 (5 actions: search→read→search→read→answer)
  练"extract value → form next query → read → answer"
- Phase 3: 2-3hop混合 (5-7 actions)
  泛化到更长链

用法：
  # 只生成2-hop demos
  python -m scripts.make_progressive_demos --split data/synth/v4_train2hop.json \
      --phase 2 --out data/sft/phase2_2hop.jsonl

  # 生成单步读取 demos（从v4全集中过滤n_gold=1的）
  python -m scripts.make_progressive_demos --split data/synth/v4_eval300.json \
      --phase 1 --out data/sft/phase1_simple.jsonl
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


def _attr_label(attr):
    return ATTRS[attr][0]


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


def build_chain_demo(inst: QAInstance, search_k: int = 10) -> list[dict]:
    """构建一个chain-following demo。

    search_k: 检索时返回的候选数，增大以避免miss gold section。
    """
    store = DocStore(inst.docs)
    env = SearchEnv(inst, store, EnvConfig(max_steps=20))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    meta = inst.meta

    if meta["kind"] != "chain_2hop" and meta["kind"] != "chain_1hop":
        return []

    ents = meta["chain_ents"]
    val_attrs = meta["chain_val_attrs"]
    link_attrs = meta["chain_link_attrs"]
    values = meta["chain_values"]

    plan: list[Action] = []
    K = len(link_attrs)  # number of hops beyond the first

    # Hop 0: search e0's a0 section
    e0, a0 = ents[0], val_attrs[0]
    q0 = f"{e0} {_attr_label(a0)}"
    hits0 = store.search(q0, k=search_k)
    idx0 = _find_hit_idx(hits0, _doc_id_for_entity(inst, e0), _attr_sid(a0))
    if not idx0:
        return []

    plan.append(Action(ActionType.SEARCH, {"query": q0}))
    plan.append(Action(ActionType.READ, {"idx": idx0}))

    # Hops 1..K: search the link value, then read the next entity's value section
    for i in range(1, len(ents)):
        Lprev = link_attrs[i - 1]
        vprev = values[i - 1]
        qL = f"{_attr_label(Lprev)} {vprev}"
        hits = store.search(qL, k=search_k)
        ei = ents[i]
        ai = val_attrs[i]

        idxL = _find_hit_idx(hits, _doc_id_for_entity(inst, ei), _attr_sid(Lprev))
        if idxL is None:
            break

        plan.append(Action(ActionType.SEARCH, {"query": qL}))
        plan.append(Action(ActionType.READ, {"doc_id": _doc_id_for_entity(inst, ei),
                                              "section_id": _attr_sid(ai)}))

    plan.append(Action(ActionType.ANSWER, {"text": inst.gold_answer, "evidence": []}))

    # Render messages with env stepping
    for a in plan:
        user_msg = _user_turn(env)
        messages.append({"role": "user", "content": user_msg})
        step = env.step(a)
        messages.append({"role": "assistant", "content": _action_text(a)})
        messages.append({"role": "tool", "content": step.observation})

    return messages


def make_simple_1hop_demos(inst: QAInstance) -> list[dict]:
    """为单属性查找问题构建最简单的demo（search→read→answer）。

    用 synth-v4 文档直接构造：找E0的A0值。
    """
    store = DocStore(inst.docs)
    eav = {}  # entity -> attr -> value
    for doc in inst.docs:
        ent = doc.title.replace(" Operations Policy", "")
        eav.setdefault(ent, {})

    meta = inst.meta
    if meta["kind"] != "chain_2hop":
        return []

    # 用2-hop问题的第一步作为简单demo
    e0 = meta["chain_ents"][0]
    a0 = meta["chain_val_attrs"][0]
    v0 = meta["chain_values"][0]

    # 构造简单问题: "What is the {a0} of {e0}?"
    simple_q = f"find the {_attr_label(a0)} of {e0}."
    simple_gold = f"{v0} {ATTRS[a0][1]}"

    # 搜索
    q0 = f"{e0} {_attr_label(a0)}"
    doc_id = _doc_id_for_entity(inst, e0)
    hits = store.search(q0, k=5)
    sid = _attr_sid(a0)
    idx = _find_hit_idx(hits, doc_id, sid)
    if not idx:
        return []

    env = SearchEnv(inst, store, EnvConfig(max_steps=20))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # search
    env.action_log = []
    env.question = simple_q
    messages.append({"role": "user", "content": _user_turn(env)})
    env.step(Action(ActionType.SEARCH, {"query": q0}))
    messages.append({"role": "assistant", "content": _action_text(Action(ActionType.SEARCH, {"query": q0}))})
    messages.append({"role": "tool", "content": env._last_observation})

    # read
    messages.append({"role": "user", "content": _user_turn(env)})
    env.step(Action(ActionType.READ, {"idx": idx}))
    messages.append({"role": "assistant", "content": _action_text(Action(ActionType.READ, {"idx": idx}))})
    messages.append({"role": "tool", "content": env._last_observation})

    # answer
    messages.append({"role": "user", "content": _user_turn(env)})
    env.step(Action(ActionType.ANSWER, {"text": simple_gold, "evidence": []}))
    messages.append({"role": "assistant", "content": _action_text(Action(ActionType.ANSWER, {"text": simple_gold, "evidence": []}))})
    messages.append({"role": "tool", "content": "ANSWER submitted: " + simple_gold})

    return messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--out", required=True)
    p.add_argument("-n", type=int, default=500)
    p.add_argument("--search-k", type=int, default=10, help="Search k for demo generation (larger avoids misses)")
    args = p.parse_args()

    from scripts.run_retriever_eval import load_split
    insts = load_split(args.split)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    k = 0
    skips = 0
    with open(args.out, "w") as f:
        for ins in insts:
            if args.phase == 1:
                # 简单demo: search→read→answer
                msgs = build_chain_demo(ins, search_k=args.search_k)
                # 只取3-action的（1-hop）
                if msgs:
                    n_actions = sum(1 for m in msgs if m["role"] == "assistant")
                    if n_actions != 3:
                        msgs = None
            elif args.phase == 2:
                # 只取5-action的（2-hop）
                msgs = build_chain_demo(ins, search_k=args.search_k)
                if msgs:
                    n_actions = sum(1 for m in msgs if m["role"] == "assistant")
                    if n_actions != 5:
                        msgs = None
            elif args.phase == 3:
                # 2-3hop混合
                msgs = build_chain_demo(ins, search_k=args.search_k)
                if msgs:
                    n_actions = sum(1 for m in msgs if m["role"] == "assistant")
                    if n_actions not in [5, 7]:
                        msgs = None

            if not msgs:
                skips += 1
                continue

            f.write(json.dumps({"instance_id": ins.instance_id, "messages": msgs},
                                ensure_ascii=False) + "\n")
            k += 1

            if k >= args.n:
                break

    print(f"wrote {k} demos -> {args.out} (skipped {skips} of first {k + skips})")


if __name__ == "__main__":
    main()