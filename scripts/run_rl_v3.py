"""REINFORCE-with-baseline RL launcher for DocScout on synth-v3 (realistic scale).

Continues training from the v3-SFT LoRA adapter (the CodeScout-style RFT
warm-start) using the read-budget reward on synth-v3 train instances.
group_size=1 => REINFORCE-with-baseline (across-instance EMA advantage), which
produces a gradient even when the SFT policy is peaked (where GRPO sees zero
within-group variance — see refine-logs/round-result.md Round 5 audit).

  python -m scripts.run_rl_v3 --base <qwen3-1.7b> \
      --lora-adapter ckpts/docscout-sft-v3-lora --reward ratio --max-steps 120 \
      --save-dir ckpts/docscout-rl-v3-ratio
"""
from __future__ import annotations

import argparse

from docscout.env.search_env import EnvConfig
from docscout.reward.reward import RewardConfig
from docscout.train.grpo import TrainConfig, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/home/linkco/.cache/modelscope/hub/models/Qwen/Qwen3-1___7B")
    p.add_argument("--lora-adapter", default="ckpts/docscout-sft-v3-lora",
                   help="SFT LoRA adapter to continue training (RFT warm-start)")
    p.add_argument("--split", default="data/synth/v3_train1k.json")
    p.add_argument("--reward", default="ratio", choices=["additive", "redundancy", "ratio"])
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--batch-instances", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--max-steps-episode", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--action-cost", type=float, default=0.0,
                   help="per read/expand action cost (Round 10 efficiency incentive)")
    p.add_argument("--w-evidence", type=float, default=0.3)
    p.add_argument("--w-gold-hit", type=float, default=0.0, help="selection bonus: any gold read")
    p.add_argument("--w-first-gold", type=float, default=0.0, help="selection bonus: first read is gold")
    p.add_argument("--save-dir", default="ckpts/docscout-rl-v3-ratio")
    args = p.parse_args()

    from scripts.run_retriever_eval import load_split
    instances = load_split(args.split)[: args.n_train]
    print(f"RL train pool: {len(instances)} synth-v3 instances ({args.split})", flush=True)

    rc = RewardConfig(gamma_ratio=1.0, action_cost=args.action_cost,
                      w_evidence=args.w_evidence, w_gold_hit=args.w_gold_hit,
                      w_first_gold=args.w_first_gold)
    tcfg = TrainConfig(
        base_model=args.base, env=EnvConfig(max_steps=args.max_steps_episode, search_k=5, rerank=False),
        reward=rc, reward_name=args.reward, lr=args.lr, group_size=1,  # REINFORCE-w-baseline
        batch_instances=args.batch_instances, max_train_steps=args.max_steps,
        max_new_tokens=96, temperature=args.temperature, save_dir=args.save_dir,
        lora_adapter=args.lora_adapter,  # continue the SFT LoRA, trainable
    )
    print(f"DocScout REINFORCE (synth-v3): base={args.base}\n"
          f"  lora_adapter={args.lora_adapter} reward={args.reward} "
          f"train={len(instances)} steps={args.max_steps} temp={args.temperature}", flush=True)
    train(instances, tcfg, args.reward, rc)


if __name__ == "__main__":
    main()
