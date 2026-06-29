"""GRPO training launcher for DocScout (GPU).

Loads env + reward configs, builds data (synth now; grounded pipeline later),
and runs `docscout.train.grpo.train`. Requires GPU + `pip install -e .[train]`.

  python -m scripts.run_train --env configs/env_default.yaml \
      --reward configs/reward_ratio.yaml --max-steps 500

CodeScout-faithful SkyRL alternative: adapt reference/codescout/src/train.py and
register docscout.reward.compute_reward as the reward hook.
"""

from __future__ import annotations

import argparse
import yaml

from docscout.data.synth_generator import generate_corpus
from docscout.env.search_env import EnvConfig
from docscout.reward.reward import RewardConfig
from docscout.train.grpo import TrainConfig, train


def _reward_cfg(path: str) -> tuple[str, RewardConfig]:
    d = yaml.safe_load(open(path))["reward"]
    name = d.pop("name")
    return name, RewardConfig(**{k: v for k, v in d.items() if hasattr(RewardConfig, k)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="configs/env_default.yaml")
    p.add_argument("--reward", default="configs/reward_ratio.yaml")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--batch-instances", type=int, default=4)
    p.add_argument("--n-instances", type=int, default=5000)
    args = p.parse_args()

    env_d = yaml.safe_load(open(args.env))
    env_cfg = EnvConfig(**env_d["env"])
    base = env_d.get("model", {}).get("base", "Qwen/Qwen3-1.7B")
    dd = env_d.get("data", {})

    instances = generate_corpus(
        n_docs=dd.get("n_docs", 8), sections_per_doc=dd.get("sections_per_doc", 8),
        n_instances=args.n_instances, multi_hop_frac=dd.get("multi_hop_frac", 0.2),
        seed=dd.get("seed", 0),
    )
    reward_name, reward_cfg = _reward_cfg(args.reward)
    tcfg = TrainConfig(
        base_model=base, env=env_cfg, reward=reward_cfg, reward_name=reward_name,
        group_size=args.group_size, batch_instances=args.batch_instances,
        max_train_steps=args.max_steps,
    )
    print(f"DocScout GRPO: {base} | reward={reward_name} | {len(instances)} instances | {args.max_steps} steps")
    train(instances, tcfg, reward_name, reward_cfg)


if __name__ == "__main__":
    main()
