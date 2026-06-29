"""GRPO RL launcher for DocScout (Phase-3). Runs from an SFT checkpoint.

Small pilot config (group/batch/steps) to show RL fixing the SFT 'reads but never
answers' degeneracy via the read-budget reward. Larger/full runs and reward
ablation V0..V4 are done by sweeping --reward and --max-steps.

  python -m scripts.run_rl --base ckpts/docscout-sft --reward ratio --max-steps 40
"""

from __future__ import annotations

import argparse

from docscout.data.synth_generator import generate_corpus
from docscout.env.search_env import EnvConfig
from docscout.reward.reward import RewardConfig
from docscout.train.grpo import TrainConfig, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="ckpts/docscout-sft")
    p.add_argument("--reward", default="ratio", choices=["additive", "redundancy", "ratio"])
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch-instances", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--max-steps-episode", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--paraphrase-frac", type=float, default=0.6)
    p.add_argument("--save-dir", default="ckpts/docscout-rl-ratio")
    p.add_argument("--lora-adapter", default=None, help="load this LoRA adapter onto base for RL")
    p.add_argument("--train-lora", action="store_true", help="add fresh LoRA to base (RL from base, no SFT)")
    args = p.parse_args()

    # v2-style training pool (middle difficulty: paraphrase + multi-hop)
    instances = generate_corpus(n_docs=16, sections_per_doc=10, n_instances=args.n_train,
                                multi_hop_frac=0.2, paraphrase_frac=args.paraphrase_frac, seed=21)
    rc = RewardConfig(gamma_ratio=1.0)
    tcfg = TrainConfig(
        base_model=args.base, env=EnvConfig(max_steps=args.max_steps_episode, search_k=5, rerank=False),
        reward=rc, reward_name=args.reward, lr=args.lr, group_size=args.group_size,
        batch_instances=args.batch_instances, max_train_steps=args.max_steps,
        max_new_tokens=96, temperature=args.temperature, save_dir=args.save_dir,
        lora_adapter=args.lora_adapter,
        train_lora=args.train_lora,
    )
    print(f"DocScout GRPO: base={args.base} reward={args.reward} "
          f"train={args.n_train} group={args.group_size} steps={args.max_steps}", flush=True)
    train(instances, tcfg, args.reward, rc)


if __name__ == "__main__":
    main()
