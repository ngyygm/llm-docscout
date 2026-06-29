"""Stratify agent eval accuracy by hop type (single vs multi-hop).

Tests the Round-9 hypothesis: read-budget RL has no leverage on single-hop
questions (1 gold section = no "how much to read" decision) but MIGHT help on
multi-hop (must read 2+ sections, decide when enough). Joins each method's eval
rows (by instance_id) with the split's meta.kind.

  python -m scripts.stratified
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path


def load_split_kinds(path="data/synth/v3_eval300.json"):
    return {r["instance_id"]: r.get("meta", {}).get("kind", "single_hop") for r in json.load(open(path))}


def stratify(eval_json, kinds):
    try:
        rows = json.load(open(eval_json))["rows"]
    except Exception:
        return None
    by = {"single_hop": [], "multi_hop": []}
    for r in rows:
        k = kinds.get(r["id"], "single_hop")
        by.setdefault(k, []).append(r["score"])
    return {k: (round(st.mean(v), 3), len(v)) for k, v in by.items() if v}


def main():
    kinds = load_split_kinds()
    methods = {
        "v3-SFT": "results/hf_agent_v3sft_greedy.json",
        "RL-ratio(s100)": "results/hf_agent_rl_v3_ratio_greedy.json",
        "RL-add(s100)": "results/hf_agent_rl_v3_add_greedy.json",
        "RL-ratio(snap s60)": "results/hf_agent_rl_v3_ratio_snap.json",
        "toy-SFT": "results/hf_agent_toysft_v3.json",
    }
    print(f"{'method':<20} {'single_hop acc (n)':<22} {'multi_hop acc (n)':<22}")
    print("-" * 64)
    out = {}
    for name, path in methods.items():
        if not Path(path).exists():
            continue
        s = stratify(path, kinds)
        if s is None:
            continue
        out[name] = s
        sh = s.get("single_hop", (0, 0))
        mh = s.get("multi_hop", (0, 0))
        print(f"{name:<20} {sh[0]:<22} {mh[0]:<22}".replace(f"{sh[0]}", f"{sh[0]:.3f} (n={sh[1]})").replace(f"{mh[0]}", f"{mh[0]:.3f} (n={mh[1]})") if False else
              f"{name:<20} {sh[0]:.3f} (n={sh[1]:<4})       {mh[0]:.3f} (n={mh[1]:<4})")
    json.dump(out, open("results/stratified.json", "w"), indent=2)
    print("\nsaved -> results/stratified.json")
    # headline: does RL beat SFT on multi-hop?
    sft = out.get("v3-SFT", {}).get("multi_hop", (None, 0))[0]
    rl = out.get("RL-ratio(s100)", {}).get("multi_hop", (None, 0))[0]
    if sft is not None and rl is not None:
        print(f"\nMulti-hop: RL-ratio {rl} vs SFT {sft} (delta {rl-sft:+.3f})")


if __name__ == "__main__":
    main()
