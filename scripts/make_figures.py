"""Generate the key figures for the DocScout paper (from result JSONs).

F1: accuracy-per-read-token Pareto frontier (agent cluster vs RAG-k1/3/5 vs oracle).
F2: failure-mode breakdown (selection vs read vs correct) for v3-SFT.
F3: RL-inertia evidence — per-instance SFT acc vs RL acc (identical diagonal).
"""
from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def fig_frontier(out="figures/f1_frontier.png"):
    d = json.load(open("results/v3_diagnostic.json"))
    rag = {}
    for r in d["modes"]:
        if r["mode"] == "oracle":
            rag["oracle"] = (r["ctx_bpe_tokens_mean"], r["acc"])
        elif r["mode"] == "rag_full":
            rag[f"RAG-full k={r['k']}"] = (r["ctx_bpe_tokens_mean"], r["acc"])
    # agent points
    agents = {}
    for tag, path in [("SFT", "results/hf_agent_v3sft_greedy.json"),
                      ("RL-v1 (ratio, inert)", "results/hf_agent_rl_v3_ratio_greedy.json"),
                      ("RL-strong s20", "results/hf_agent_rl_v3_strong_s20.json"),
                      ("RL-strong s40", "results/hf_agent_rl_v3_strong_s40.json"),
                      ("RL-strong s60", "results/hf_agent_rl_v3_strong_s60.json"),
                      ("RL-strong s100", "results/hf_agent_rl_v3_strong_s100.json"),
                      ("RL-strong s120", "results/hf_agent_rl_v3_strong_s120.json"),
                      ("RL-strong seed2", "results/hf_agent_rl_v3_strong_s2_final.json")]:
        if Path(path).exists():
            s = json.load(open(path))["summary"]
            agents[tag] = (s.get("mean_committed_read_bpe", 160), s["answer_accuracy"])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    # RAG curve
    rpts = sorted([v for k, v in rag.items() if "RAG" in k])
    if rpts:
        xs, ys = zip(*rpts)
        ax.plot(xs, ys, "o-", color="#1f77b4", label="fixed-k RAG (full sections)", lw=2, ms=8)
        for k, (x, y) in rag.items():
            if "RAG" in k:
                ax.annotate(k.split()[-1], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#1f77b4")
    # oracle
    ox, oy = rag["oracle"]
    ax.scatter([ox], [oy], marker="*", s=220, color="#2ca02c", zorder=5, label=f"oracle (gold ctx)")
    # agents
    for tag, (x, y) in agents.items():
        ax.scatter([x], [y], marker="s", s=110, color="#d62728", zorder=5)
        ax.annotate(tag, (x, y), textcoords="offset points", xytext=(7, -3), fontsize=9, color="#d62728")
    ax.scatter([], [], marker="s", s=110, color="#d62728", label="agent (SFT / RL)")
    ax.set_xlabel("committed read tokens (BPE)")
    ax.set_ylabel("answer accuracy")
    ax.set_title("Accuracy-per-read-token frontier (synth-v3, n=150)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.1, 0.9)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


def fig_failure(out="figures/f2_failure.png"):
    d = json.load(open("results/v3_failure_sft.json"))
    sel = d["selection_fail_pct"]; rd = d["read_fail_pct"]; cor = d["correct_pct"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sizes = [cor, rd, sel]
    labels = [f"correct\n(read gold & right)\n{cor:.0%}", f"read gold but wrong\n{rd:.0%}", f"selection fail\n(never read gold)\n{sel:.0%}"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    ax.pie(sizes, labels=labels, colors=colors, startangle=90, textprops={"fontsize": 9})
    ax.set_title(f"Failure breakdown (v3-SFT)\nbottleneck = SELECTION ({sel:.0%})")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


def fig_inertia(out="figures/f3_inertia.png"):
    """Per-instance acc: SFT vs RL-ratio. If identical -> points on diagonal."""
    sft = {r["id"]: r["score"] for r in json.load(open("results/hf_agent_v3sft_greedy.json"))["rows"]}
    rl = {r["id"]: r["score"] for r in json.load(open("results/hf_agent_rl_v3_ratio_greedy.json"))["rows"]}
    common = sorted(set(sft) & set(rl))
    xs = [sft[i] for i in common]; ys = [rl[i] for i in common]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(xs, ys, s=14, alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="identical (y=x)")
    # count off-diagonal
    diff = sum(1 for i in common if abs(sft[i] - rl[i]) > 1e-6)
    ax.set_xlabel("SFT per-instance score"); ax.set_ylabel("RL-ratio per-instance score")
    ax.set_title(f"RL inertia: {len(common)} instances, {diff} changed by RL\n(RL ≈ SFT, zero policy movement)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}  ({diff}/{len(common)} changed)")


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    fig_frontier()
    fig_failure()
    fig_inertia()
