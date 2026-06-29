"""Build the accuracy-per-read-token frontier for DocScout from result JSONs.

Collects (answer_accuracy, mean committed-read BPE) for every method, sorts by
cost, and reports the Pareto frontier + which method dominates at each cost.
This is the headline figure of the paper (claim C1: read-budget policy dominates
fixed-k RAG on the accuracy-per-read-token frontier).

Reads:
  - results/v3_diagnostic.json  (oracle, rag_full k1/3/5, rag_snip — ctx_bpe_tokens_mean)
  - results/hf_agent_*.json     (SFT / RL agents — summary.mean_committed_read_bpe)
Emits results/frontier.json + a printed table.

  python -m scripts.frontier
"""
from __future__ import annotations

import json
from pathlib import Path


def load_diag():
    d = json.load(open("results/v3_diagnostic.json"))
    out = {}
    for r in d["modes"]:
        if r["mode"] == "oracle":
            out["oracle"] = (r["acc"], r["ctx_bpe_tokens_mean"])
        elif r["mode"] == "rag_full":
            out[f"rag_full_k{r['k']}"] = (r["acc"], r["ctx_bpe_tokens_mean"])
        elif r["mode"] == "rag_snip":
            out[f"rag_snip_k{r['k']}"] = (r["acc"], r["ctx_bpe_tokens_mean"])
    return out


def load_agents():
    """Pull agent points from hf_agent_*.json (tag -> (acc, committed_bpe, n_read))."""
    out = {}
    for p in sorted(Path("results").glob("hf_agent_*v3*.json")):
        try:
            s = json.load(open(p))["summary"]
            tag = s.get("tag", p.stem)
            acc = s["answer_accuracy"]
            bpe = s.get("mean_committed_read_bpe") or s.get("mean_committed_read_tokens", 0)
            out[tag] = (acc, round(bpe), s.get("mean_n_read", 0))
        except Exception:
            pass
    return out


def pareto(points):
    """points: dict name->(acc, cost). Return Pareto-optimal names (max acc, min cost)."""
    items = sorted(points.items(), key=lambda kv: (kv[1][1], -kv[1][0]))  # cost asc, acc desc
    front, best_acc = [], -1
    for name, (acc, cost) in items:
        if acc > best_acc:
            front.append(name)
            best_acc = acc
    return front


def main():
    diag = load_diag()
    agents = load_agents()
    # unify into (acc, cost_bpe) — drop n_read for frontier
    pts = {k: (v[0], v[1]) for k, v in diag.items()}
    agent_pts = {k: (v[0], v[1]) for k, v in agents.items()}
    all_pts = {**pts, **agent_pts}

    print("=" * 70)
    print("DocScout accuracy-per-read-token frontier (synth-v3, n=150)")
    print("=" * 70)
    print(f"{'method':<22} {'acc':>7} {'read_BPE':>9} {'acc/kBPE':>9}")
    print("-" * 70)
    for name in sorted(all_pts, key=lambda n: all_pts[n][1]):
        acc, cost = all_pts[name]
        print(f"{name:<22} {acc:>7.3f} {cost:>9.0f} {acc/max(cost,1)*1000:>8.2f}")

    front = pareto(all_pts)
    print("\nPareto-optimal methods:", front)

    # headline comparison: best agent vs rag_full_k5 (the strong RAG baseline)
    rag5 = pts.get("rag_full_k5")
    best_agent = max(agent_pts.items(), key=lambda kv: kv[1][0]) if agent_pts else None
    if rag5 and best_agent:
        an, (aa, ac) = best_agent
        ra, rc = rag5
        print(f"\nBest agent {an}: acc={aa:.3f} @ {ac:.0f} BPE")
        print(f"RAG-full k5      : acc={ra:.3f} @ {rc:.0f} BPE")
        if aa >= ra - 0.005:
            print(f"  -> agent matches RAG accuracy at {ac/rc:.1%} of the cost "
                  f"(~{rc/ac:.1f}x fewer read tokens)")
        else:
            print(f"  -> agent acc gap vs RAG5: {ra-aa:+.3f} (RAG still higher accuracy)")

    json.dump({"points": {k: {"acc": v[0], "read_bpe": v[1]} for k, v in all_pts.items()},
               "pareto": front},
              open("results/frontier.json", "w"), indent=2)
    print("\nsaved -> results/frontier.json")


if __name__ == "__main__":
    main()
