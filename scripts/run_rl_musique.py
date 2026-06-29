"""REINFORCE-with-baseline RL launcher for DocScout on REAL MuSiQue data.

Trains from an SFT checkpoint using the read-budget reward on held-out-disjoint
MuSiQue train instances. Temperature 1.0 for exploration (so the policy produces
varied read/skip trajectories RL can learn from), max_steps 8 so the agent CAN
read multiple sections when the answer needs it.

  python -m scripts.run_rl_musique --base ckpts/docscout-sft12-1b5-musique-fixed \
      --reward ratio --max-steps 60 --save-dir ckpts/docscout-rl-musique-ratio
"""
from __future__ import annotations

import argparse

from docscout.env.search_env import EnvConfig
from docscout.reward.reward import RewardConfig
from docscout.train.grpo import TrainConfig, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--split", default="data/grounded/musique_train.json")
    p.add_argument("--reward", default="ratio", choices=["additive", "redundancy", "ratio"])
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--batch-instances", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--max-steps-episode", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--save-dir", default="ckpts/docscout-rl-musique-ratio")
    args = p.parse_args()

    from scripts.run_retriever_eval import load_split
    # use train instances DISJOINT from eval (musique_test). Offset past the SFT
    # demos (which used train[0:800]) to give RL fresh-ish instances.
    all_train = load_split(args.split)
    instances = all_train[800: 800 + args.n_train]  # train[800:1100], unseen by SFT demos
    print(f"RL train pool: {len(instances)} MuSiQue instances (train[800:{800+args.n_train}])", flush=True)

    rc = RewardConfig(gamma_ratio=1.0)
    tcfg = TrainConfig(
        base_model=args.base, env=EnvConfig(max_steps=args.max_steps_episode, search_k=5, rerank=False),
        reward=rc, reward_name=args.reward, lr=args.lr, group_size=1,
        batch_instances=args.batch_instances, max_train_steps=args.max_steps,
        max_new_tokens=96, temperature=args.temperature, save_dir=args.save_dir,
    )
    print(f"DocScout REINFORCE (MuSiQue): base={args.base} reward={args.reward} "
          f"train={len(instances)} steps={args.max_steps} temp={args.temperature}", flush=True)
    train(instances, tcfg, args.reward, rc)


if __name__ == "__main__":
    main()
