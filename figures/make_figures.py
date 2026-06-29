"""Generate paper figures from real result JSONs."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path("results")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150})

# ---- F1: baselines on synth-v2 (direct vs paraphrased) ----
orc = json.load(open(R/"v2_oracle_vllm.json"))
rag = json.load(open(R/"v2_rag_vllm.json"))
prm = json.load(open(R/"v2_prompted.json"))
methods = ["Oracle", "One-shot RAG", "Prompted\n(no train)"]
overall = [orc["answer_acc"], rag["answer_acc"], prm["answer_acc"]]
direct = [orc["acc_direct"], rag["acc_direct"], prm["answer_acc"]]      # prompted: no split -> overall
para = [orc["acc_paraphrased"], rag["acc_paraphrased"], prm["answer_acc"]]
import numpy as np
x = np.arange(len(methods)); w = .35
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.bar(x - w/2, direct, w, label="direct query", color="#4C72B0")
ax.bar(x + w/2, para, w, label="paraphrased query", color="#DD8452")
ax.set_ylabel("Answer accuracy"); ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylim(0, 1.0); ax.set_title("Phase-2 baselines (synth-v2, n=400/200)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig("figures/f1_baselines.pdf"); fig.savefig("figures/f1_baselines.png")
print("F1 done")

# ---- F2: SFT stop-failure -> fix ----
v1 = json.load(open(R/"rl_signal.json"))       # SFT-v1 (plain)
v2 = json.load(open(R/"rl_signal_sft2.json"))  # SFT-v2 (answer-balanced)
labels = ["Answer rate", "Correct rate", "Group-reward\nstd (advantage)"]
v1v = [v1["answer_rate"], v1.get("correct_rate", 0.0), v1["group_reward_std"]]
v2v = [v2["answer_rate"], v2["correct_rate"], v2["group_reward_std"]]
x = np.arange(len(labels)); w = .35
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.bar(x - w/2, v1v, w, label="SFT-v1 (plain)", color="#C44E52")
ax.bar(x + w/2, v2v, w, label="SFT-v2 (answer-balanced)", color="#55A868")
ax.set_ylabel("Value"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("SFT stop-failure pathology and fix")
ax.legend(fontsize=8, loc="upper left")
for xi, vv in zip(x - w/2, v1v): ax.text(xi, vv+.01, f"{vv:.2f}", ha="center", fontsize=7)
for xi, vv in zip(x + w/2, v2v): ax.text(xi, vv+.01, f"{vv:.2f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig("figures/f2_sft_fix.pdf"); fig.savefig("figures/f2_sft_fix.png")
print("F2 done")
