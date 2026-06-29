"""Fast RL-signal probe via vLLM (Phase-3 prerequisite check).

The SFT agent is *greedy*-degenerate (never answers). RL needs positive-reward
trajectories to exist in the sampling distribution. This samples G rollouts per
instance at high temperature via vLLM (fast, batched) and reports the reward
distribution + answer rate — i.e. whether RL has a learnable signal. Also feeds
a one-shot REINFORCE-style update is possible downstream.

  python -m scripts.vllm_rollout_collect -n 60 --group 8
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor

from docscout.agent.client import OpenAIClient
from docscout.agent.rollout import rollout
from docscout.env.search_env import EnvConfig
from docscout.reward.answer_scoring import score_answer


def _load(path):
    from scripts.run_retriever_eval import load_split
    return load_split(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/synth/v2_dev400.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("-n", type=int, default=60)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="results/rl_signal.json")
    args = p.parse_args()

    insts = _load(args.split)[: args.n]
    client = OpenAIClient(model="x", base_url=args.base_url, temperature=args.temp, max_tokens=96)
    import urllib.request
    client.model = json.loads(urllib.request.urlopen(args.base_url + "/models", timeout=5).read())["data"][0]["id"]

    def one(ins):
        ress = []
        for _ in range(args.group):
            res = rollout(ins, client, env_config=EnvConfig(max_steps=args.max_steps, search_k=5, rerank=False),
                          reward_name="ratio")
            s = score_answer(res.trajectory.final_answer, ins.gold_answer)
            for a in ins.meta.get("answer_aliases", []):
                s = max(s, score_answer(res.trajectory.final_answer, a))
            ress.append((res.reward, res.components, s, res.trajectory.terminated_by == "answer"))
        return ress

    all_res = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ress in ex.map(one, insts):
            all_res.append(ress)

    flat = [r for ress in all_res for r in ress]
    rewards = [r[0] for r in flat]
    answered = [r[3] for r in flat]
    correct = [r[2] > 0 for r in flat]
    reads = [r[1]["read_tokens"] for r in flat]
    # per-instance: does at least one rollout answer / answer correctly? (GRPO group signal)
    any_ans = sum(1 for ress in all_res if any(r[3] for r in ress))
    any_corr = sum(1 for ress in all_res if any(r[2] > 0 for r in ress))
    # reward spread (GRPO advantage signal)
    grp_std = st.mean(st.pstdev([r[0] for r in ress]) for ress in all_res if len(ress) > 1)

    print(f"=== RL signal probe (n_inst={len(insts)}, G={args.group}, temp={args.temp}) ===")
    print(f"  rollouts answered   : {sum(answered)}/{len(flat)} = {st.mean(answered):.3f}")
    print(f"  rollouts correct    : {sum(correct)}/{len(flat)} = {st.mean(correct):.3f}")
    print(f"  mean reward         : {st.mean(rewards):+.3f}  (std {st.pstdev(rewards):.3f})")
    print(f"  mean read-tokens    : {st.mean(reads):.0f}")
    print(f"  instances w/ >=1 answering rollout  : {any_ans}/{len(insts)} = {any_ans/len(insts):.3f}")
    print(f"  instances w/ >=1 CORRECT rollout    : {any_corr}/{len(insts)} = {any_corr/len(insts):.3f}")
    print(f"  group reward std (advantage signal) : {grp_std:.3f}")
    if args.out:
        json.dump({"n_inst": len(insts), "group": args.group, "temp": args.temp,
                   "answer_rate": st.mean(answered), "correct_rate": st.mean(correct),
                   "mean_reward": st.mean(rewards), "reward_std": st.pstdev(rewards),
                   "inst_with_correct_rollout": any_corr / len(insts),
                   "group_reward_std": grp_std}, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
