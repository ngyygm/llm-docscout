"""Unit tests for the DocScout core (run with `python -m pytest tests/` or `python tests/test_core.py`)."""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docscout.data.synth_generator import generate_corpus
from docscout.env.docstore import DocStore
from docscout.env.search_env import EnvConfig, SearchEnv
from docscout.agent.parsing import parse_action
from docscout.reward.reward import RewardConfig, compute_reward
from docscout.types import Action, ActionType


def _make_env():
    insts = generate_corpus(n_docs=4, sections_per_doc=5, n_instances=1, multi_hop_frac=0.0, seed=42)
    inst = insts[0]
    store = DocStore(inst.docs)
    return inst, SearchEnv(inst, store, EnvConfig(max_steps=8, max_read_tokens=2000, search_k=4))


def test_synth_has_gold_evidence():
    insts = generate_corpus(n_docs=4, sections_per_doc=5, n_instances=10, seed=1)
    assert len(insts) == 10
    for ins in insts:
        assert ins.gold_answer
        assert ins.gold_evidence, "every instance must have section-locatable gold evidence"
        for ev in ins.gold_evidence:
            uids = {(d.doc_id, s.section_id) for d in ins.docs for s in d.sections}
            assert (ev.doc_id, ev.section_id) in uids, "gold evidence must point to a real section"


def test_docstore_search_read_expand():
    inst, _ = _make_env()
    store = DocStore(inst.docs)
    hits = store.search("synchronize approval", k=3)
    assert hits and len(hits) <= 3
    assert "snippet" in hits[0] and "content" not in hits[0]  # snippet-only
    h = hits[0]
    r = store.read(h["doc_id"], h["section_id"])
    assert r and "content" in r and r["token_len"] > 0
    # expand right should give a neighbor or None (no crash)
    nbr = store.expand(h["doc_id"], h["section_id"], "right")
    assert nbr is None or ("content" in nbr and nbr["section_id"] != h["section_id"])
    assert store.expand(h["doc_id"], h["section_id"], "sideways") is None  # bad direction


def test_env_budget_and_answer():
    inst, env = _make_env()
    env.step(Action(ActionType.SEARCH, {"query": inst.question}))
    assert env.n_search == 1 and not env.done
    # exhaust step budget without answering
    for _ in range(env.cfg.max_steps + 1):
        if env.done:
            break
        env.step(Action(ActionType.SEARCH, {"query": "x"}))
    assert env.done and env.terminated_by in ("step_budget", "token_budget")


def test_env_committed_vs_snippet():
    inst, env = _make_env()
    env.step(Action(ActionType.SEARCH, {"query": inst.question}))
    assert len(env.read_uids()) > 0, "search snippets enter context"
    assert len(env.committed_read_uids()) == 0, "no committed read yet"
    # read the first candidate
    h = env.current_candidates[0]
    env.step(Action(ActionType.READ, {"doc_id": h["doc_id"], "section_id": h["section_id"]}))
    assert len(env.committed_read_uids()) == 1


def test_reward_variants_differ_on_waste():
    inst, env = _make_env()
    # search + read several non-gold sections (wasteful) then answer wrong
    env.step(Action(ActionType.SEARCH, {"query": inst.question}))
    for h in env.current_candidates[:3]:
        env.step(Action(ActionType.READ, {"doc_id": h["doc_id"], "section_id": h["section_id"]}))
    env.step(Action(ActionType.ANSWER, {"text": "wrong", "evidence": []}))
    cfg = RewardConfig()
    r_add, c_add = compute_reward("additive", env, inst, cfg)
    r_red, c_red = compute_reward("redundancy", env, inst, cfg)
    assert c_red["redundancy"] >= 0
    assert r_red <= r_add, "redundancy reward must be <= additive when redundancy>0"


def test_reward_ratio_rewards_efficient_gold_read():
    inst, env = _make_env()
    # efficient: search then read ONLY the gold section then answer correctly
    env.step(Action(ActionType.SEARCH, {"query": inst.question}))
    g = inst.gold_evidence[0]
    env.step(Action(ActionType.READ, {"doc_id": g.doc_id, "section_id": g.section_id}))
    env.step(Action(ActionType.ANSWER, {"text": inst.gold_answer,
                                        "evidence": [{"doc_id": g.doc_id, "section_id": g.section_id}]}))
    _, c = compute_reward("ratio", env, inst, RewardConfig())
    assert c["answer"] == 1.0
    assert c["evidence"] > 0
    assert c["efficiency_ratio"] > 0


def test_parse_action_formats():
    a = parse_action("blah\nACTION: read\ndoc_id: doc_01\nsection_id: 3\n")
    assert a.type == ActionType.READ and a.args["doc_id"] == "doc_01"
    a = parse_action("ACTION: answer\ntext: 5 min\nevidence: doc_01:3, doc_02:1\n")
    assert a.type == ActionType.ANSWER and a.args["text"] == "5 min" and len(a.args["evidence"]) == 2
    a = parse_action("garbage")
    assert a.type == ActionType.SEARCH  # graceful fallback


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return 0 if passed == len(fns) else 1


if __name__ == "__main__":
    sys.exit(main())
