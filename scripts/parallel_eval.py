"""Multi-GPU parallel eval launcher for DocScout diagnostics/baselines.

Data-parallel: shard instances across N GPUs, one process per GPU (each loads its
own model copy), then merge. ~N× speedup. All 3 local RTX 3090s used by default.

  python -m scripts.parallel_eval --mode oracle -n 300
  python -m scripts.parallel_eval --mode oneshot_rag -n 300
  python -m scripts.parallel_eval --mode prompted -n 150 --max-steps 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="oracle", choices=["oracle", "oneshot_rag", "prompted"])
    p.add_argument("--modes", default=None, help="comma-separated modes to run in sequence, e.g. oracle,oneshot_rag,prompted")
    p.add_argument("--split", default="data/grounded/musique_dev.json")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("-n", type=int, default=300)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--num-gpus", type=int, default=3)
    p.add_argument("--tag", default=None, help="output name; default = <mode>")
    args = p.parse_args()
    modes = args.modes.split(",") if args.modes else [args.mode]
    summary = {}
    for mode in modes:
        summary[mode] = run_one(mode, args)
    print("\n=== BATTERY SUMMARY ===")
    for mode, acc in summary.items():
        print(f"  {mode:12s} answer_acc = {acc:.3f}")
    json.dump(summary, open(Path("results") / "battery_summary.json", "w"), indent=2)


def run_one(mode, args):
    tag = args.tag or mode
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    procs = []
    shard_files = []
    for g in range(args.num_gpus):
        sf = out_dir / f"{tag}_shard{g}.json"
        shard_files.append(sf)
        cmd = [sys.executable, "-m", "scripts.run_oracle_eval", "--backend", "hf",
               "--mode", mode, "--split", args.split, "--model", args.model,
               "-n", str(args.n), "--k", str(args.k),
               "--shard", str(g), "--num-shards", str(args.num_gpus), "--out", str(sf)]
        env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(g),
               "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        print(f"[{mode} gpu {g}] shard {g}/{args.num_gpus} -> {sf}")
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(f"logs/eval_{tag}_gpu{g}.log", "w"),
                                      stderr=subprocess.STDOUT))
    for proc in procs:
        proc.wait()
    total_n, acc_sum = 0, 0.0
    for sf in shard_files:
        if sf.exists():
            d = json.load(open(sf))
            total_n += d["n"]
            acc_sum += d["answer_acc"] * d["n"]
    acc = (acc_sum / total_n) if total_n else 0.0
    json.dump({"mode": mode, "model": args.model, "n": total_n, "answer_acc": acc},
              open(out_dir / f"{tag}.json", "w"), indent=2)
    print(f"  [{mode}] n={total_n} answer_acc={acc:.3f}")
    return acc


if __name__ == "__main__":
    main()
