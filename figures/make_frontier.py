"""Regenerate the accuracy-per-read-token frontier (f3) with the POSITIVE RL result.

Points (Qwen2.5-7B-Instruct, synth-v2 held-out seed77, n=150, leakage-free):
  SFT            : 0.660 acc @ 37 committed read-tok
  SFT (greedy)   : 0.673 acc @ 38 tok
  RL REINFORCE-ratio (120 steps): 0.713 @ 40 tok
  RL REINFORCE-add  (120 steps): 0.727 @ 40 tok
  RAG (top-5)    : 0.960 @ ~380 tok (5 snippets)
  Oracle (gold)  : 0.940 @ gold context
RL Pareto-dominates SFT in the low-read-budget regime (+0.05-0.07 acc, ~same tok).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, committed_read_tokens, accuracy, marker, color)
pts = [
    ("SFT",            37.1, 0.660, "s", "#4C72B0"),
    ("SFT (greedy)",   37.8, 0.673, "s", "#4C72B0"),
    ("RL-ratio",       39.9, 0.713, "^", "#DD8452"),
    ("RL-add",         39.9, 0.727, "^", "#C44E52"),
    ("RAG (top-5)",    380,  0.960, "D", "#55A868"),
    ("Oracle",         520,  0.940, "*", "#8172B3"),
]

fig, ax = plt.subplots(figsize=(5.0, 3.4))
# shade the low-read-budget regime where the agent lives
ax.axvspan(0, 80, color="#FFF3CD", alpha=0.5, zorder=0, label="read-budget regime")
for label, tok, acc, mk, col in pts:
    is_rl = label.startswith("RL")
    is_sft = label.startswith("SFT")
    ax.scatter(tok, acc, s=190 if "Oracle" in label else 130, marker=mk,
               color=col, edgecolor="black", linewidth=0.8, zorder=3)
    # label placement
    dx, dy = (6, -0.012) if "RAG" in label else (6, 0.006)
    if label == "Oracle": dx, dy = (8, -0.02)
    if label == "SFT (greedy)": dx, dy = (6, -0.022)
    ax.annotate(label, (tok, acc), xytext=(tok + dx, acc + dy), fontsize=8.5, zorder=4)
# connect SFT -> RL to show the RL gain at ~constant cost
ax.annotate("", xy=(39.9, 0.713), xytext=(37.1, 0.660),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.6))
ax.text(41, 0.685, "RL > SFT\n(+0.05 acc,\n~same tok)", fontsize=7.5,
        color="#C44E52", ha="left")

ax.set_xlabel("committed read-tokens (document content read into context)")
ax.set_ylabel("answer accuracy")
ax.set_xlim(-15, 600)
ax.set_ylim(0.60, 0.99)
ax.set_title("Accuracy-per-read-token frontier (held-out n=150, leakage-free)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/f3_frontier_clean.pdf")
fig.savefig("figures/f3_frontier_clean.png", dpi=160)
print("saved figures/f3_frontier_clean.{pdf,png}")
