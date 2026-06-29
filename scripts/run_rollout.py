"""R0 smoke test: run DocScout rollouts end-to-end on CPU with the StubClient.

Verifies the full plumbing (synth data -> DocStore -> SearchEnv -> rollout ->
reward engine -> metrics) and that the reward (a) is computed, (b) distinguishes
reward variants, (c) tracks read-budget metrics. No model, no GPU.

Usage:
    python -m scripts.run_rollout --n-instances 50 --reward ratio
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict

from docscout.agent.client import StubClient
from docscout.agent.rollout import rollout
from docscout.data.synth_generator import generate_corpus
from docscout.env.search_env import EnvConfig
from docscout.reward.reward import RewardConfig


def run(n_instances: int, seed: int, reward_name: str, max_steps: int, verbose: bool, out: str | None):
    instances = generate_corpus(n_docs=8, sections_per_doc=8, n_instances=n_instances,
                                multi_hop_frac=0.2, seed=seed)
    client = StubClient(max_reads=3)
    env_cfg = EnvConfig(max_steps=max_steps, max_read_tokens=2000, search_k=5)

    # collect under all three variants for comparison
    variants = ["additive", "redundancy", "ratio"]
    results = rollout_batch(instances, client, env_cfg, variants)
    summary = _summarize(results, reward_name, max_steps, n_instances)
    print(f"\n=== R0 smoke: {len(instances)} instances, reward={reward_name}, max_steps={max_steps} ===")
    for v in variants:
        rs = results[v]
        rewards = [r.reward for r in rs]
        ans = [r.components["answer"] for r in rs]
        rtok = [r.components["read_tokens"] for r in rs]
        eff = [r.components["efficiency_ratio"] for r in rs]
        nsub = sum(1 for r in rs if r.components["terminated_by"] == "answer")
        print(f"[{v}]  mean_reward={st.mean(rewards):+.3f}  "
              f"answer={st.mean(ans):.3f}  read_tok={st.mean(rtok):.0f}  "
              f"eff_ratio={st.mean(eff):.3f}  submitted={nsub}/{len(rs)}")

    # frontier-ish: bucket by read_tokens, show mean answer score per bucket (ratio reward)
    print("=== accuracy-per-read-token buckets (stub, ratio reward) ===")
    buckets = defaultdict(list)
    for r in results["ratio"]:
        b = (r.components["read_tokens"] // 100) * 100
        buckets[b].append(r.components["answer"])
    for b in sorted(buckets):
        print(f"  read_tokens [{b:4d},{b+100:4d}): n={len(buckets[b]):2d}  mean_answer={st.mean(buckets[b]):.3f}")

    if verbose and results["ratio"]:
        r = results["ratio"][0]
        t = r.trajectory
        print(f"=== example trajectory {t.instance_id} ===")
        print("Q:", instances[0].question)
        print("gold:", instances[0].gold_answer, "| gold ev:", [(e.doc_id, e.section_id) for e in instances[0].gold_evidence])
        for a in t.actions:
            print("  -", a.to_log())
        print("  final_answer:", t.final_answer, "| read_tokens:", t.total_read_tokens,
              "| terminated_by:", t.terminated_by)

    if out:
        import json, pathlib
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(out, "w"), indent=2)
        print(f"saved summary -> {out}")


def _summarize(results, reward_name, max_steps, n_instances):
    import statistics as st
    s = {"reward_name": reward_name, "max_steps": max_steps, "n_instances": n_instances, "variants": {}}
    for v, rs in results.items():
        s["variants"][v] = {
            "mean_reward": st.mean(r.reward for r in rs),
            "mean_answer": st.mean(r.components["answer"] for r in rs),
            "mean_read_tokens": st.mean(r.components["read_tokens"] for r in rs),
            "mean_efficiency_ratio": st.mean(r.components["efficiency_ratio"] for r in rs),
            "submitted": sum(1 for r in rs if r.components["terminated_by"] == "answer"),
        }
    return s


def rollout_batch(instances, client, env_cfg, variants):
    out = {v: [] for v in variants}
    for ins in instances:
        for v in variants:
            out[v].append(rollout(ins, client, env_config=env_cfg, reward_name=v))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-instances", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward", type=str, default="ratio", choices=["additive", "redundancy", "ratio"])
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=str, default=None, help="path to save summary JSON")
    args = p.parse_args()
    run(args.n_instances, args.seed, args.reward, args.max_steps, args.verbose, args.out)


if __name__ == "__main__":
    main()
