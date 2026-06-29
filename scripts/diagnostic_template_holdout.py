"""Diagnostic: template holdout analysis for v4 chain data.

Extracts chain_template categories (by chain_val_attrs + chain_link_attrs tuple)
from v4_train1k, then re-evaluates on most-common vs least-common overlapping
templates, and runs two holdout experiments:

  (a) Holdout by chain length — evaluate whether the model (trained on all hops)
      can complete 4-hop/5-hop chains. Compares model trained on short chains
      only (1-3 hop) vs short+long to see if there is evidence of long-chain
      template memorisation.

  (b) Holdout by link-attribute sequence — evaluate on instances whose link_attrs
      start with a specific prefix that also appears in training (makes the
      template-reuse test meaningful), then compare against the baseline.

If performance collapses, the model relies on template memorization rather than
a general search strategy.

Uses the existing HFClient + rollout infrastructure — no pipeline rewrite.

Usage:
  python -m scripts.diagnostic_template_holdout \
      --model ckpts/docscout-sft \
      --device cuda:0 \
      --n 100 \
      --out results/diagnostic/template_holdout.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from docscout.agent.client import HFClient
from docscout.agent.rollout import rollout
from docscout.env.search_env import EnvConfig
from docscout.reward.answer_scoring import score_answer
from docscout.types import Document, EvidenceSpan, QAInstance, Section

# ---------------------------------------------------------------------------
# Data loading (reused from run_retriever_eval + diagnostic_snippet_only)
# ---------------------------------------------------------------------------


def load_split(path: str) -> list[QAInstance]:
    raw = json.load(open(path))
    out = []
    for r in raw:
        docs = [
            Document(
                doc_id=d["doc_id"],
                title=d["title"],
                sections=[
                    Section(
                        section_id=s["section_id"],
                        title=s["title"],
                        text=s["text"],
                    )
                    for s in d["sections"]
                ],
            )
            for d in r["docs"]
        ]
        gold = [
            EvidenceSpan(doc_id=e["doc_id"], section_id=e["section_id"])
            for e in r["gold_evidence"]
        ]
        out.append(
            QAInstance(
                instance_id=r["instance_id"],
                question=r["question"],
                gold_answer=str(r["gold_answer"]),
                gold_evidence=gold,
                docs=docs,
                meta=r.get("meta", {}),
            )
        )
    return out


def _template_key(inst: QAInstance) -> tuple:
    """Full template identity: (chain_val_attrs, chain_link_attrs)."""
    m = inst.meta or {}
    return (tuple(m.get("chain_val_attrs", [])), tuple(m.get("chain_link_attrs", [])))


def _link_prefix(inst: QAInstance, n: int = 2) -> tuple:
    """First n link attributes — used for the sequence holdout."""
    m = inst.meta or {}
    return tuple(m.get("chain_link_attrs", []))[:n]


def _hop_label(inst: QAInstance) -> str:
    m = inst.meta or {}
    return f"{m.get('K', 0) + 1}-hop"


# ---------------------------------------------------------------------------
# Template distribution analysis
# ---------------------------------------------------------------------------


def analyze_templates(train_insts: list[QAInstance], eval_insts: list[QAInstance]) -> dict:
    """Build train template distribution and find overlap with eval.

    Returns the most-frequent and least-frequent templates that appear in BOTH
    train and eval (so we can actually evaluate them).
    """
    train_counts = Counter()
    for ins in train_insts:
        train_counts[_template_key(ins)] += 1

    eval_counts = Counter()
    for ins in eval_insts:
        eval_counts[_template_key(ins)] += 1

    overlap = set(train_counts) & set(eval_counts)
    sorted_overlap = sorted(overlap, key=lambda k: train_counts[k], reverse=True)

    most_key = sorted_overlap[0] if sorted_overlap else None
    least_key = sorted_overlap[-1] if sorted_overlap else None

    total_train = sum(train_counts.values())
    singleton = sum(1 for _, c in train_counts.items() if c == 1)

    return {
        "total_train_templates": len(train_counts),
        "total_eval_templates": len(eval_counts),
        "overlap_templates": len(overlap),
        "most_frequent_overlap": (
            {
                "val_attrs": list(most_key[0]),
                "link_attrs": list(most_key[1]),
                "train_count": train_counts[most_key],
                "eval_count": eval_counts[most_key],
            }
            if most_key
            else None
        ),
        "least_frequent_overlap": (
            {
                "val_attrs": list(least_key[0]),
                "link_attrs": list(least_key[1]),
                "train_count": train_counts[least_key],
                "eval_count": eval_counts[least_key],
            }
            if least_key
            else None
        ),
        "distribution_summary": {
            "total_train_instances": total_train,
            "singleton_templates": singleton,
            "mean_train_count": round(total_train / max(len(train_counts), 1), 2),
            "median_train_count": sorted(train_counts.values())[len(train_counts) // 2] if train_counts else 0,
        },
    }


# ---------------------------------------------------------------------------
# Instance selectors
# ---------------------------------------------------------------------------


def by_template(insts: list[QAInstance], key: tuple) -> list[QAInstance]:
    return [ins for ins in insts if _template_key(ins) == key]


def by_link_prefix(insts: list[QAInstance], prefix: tuple) -> list[QAInstance]:
    return [ins for ins in insts if _link_prefix(ins, len(prefix)) == prefix]


def by_chain_length(insts: list[QAInstance], max_k: int) -> list[QAInstance]:
    """Instances with K <= max_k (hop = K+1)."""
    return [ins for ins in insts if (ins.meta or {}).get("K", 0) <= max_k]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    client: HFClient,
    insts: list[QAInstance],
    env_cfg: EnvConfig,
    tag: str = "",
    verbose: bool = True,
) -> dict:
    """Rollout-based agent eval on a list of instances."""
    if not insts:
        return {
            "tag": tag,
            "n": 0,
            "accuracy": 0.0,
            "answer_rate": 0.0,
            "mean_n_read": 0.0,
            "mean_n_steps": 0.0,
            "rows": [],
            "stratified_by_hop": [],
        }

    rows = []
    n_answered = 0
    for i, ins in enumerate(insts):
        res, env = rollout(ins, client, env_config=env_cfg, return_env=True)
        sc = score_answer(res.trajectory.final_answer, str(ins.gold_answer))
        rows.append(
            {
                "instance_id": ins.instance_id,
                "gold": str(ins.gold_answer),
                "pred": res.trajectory.final_answer[:80],
                "score": sc,
                "n_search": res.trajectory.n_search,
                "n_read": res.trajectory.n_read,
                "n_expand": res.trajectory.n_expand,
                "n_steps": res.trajectory.n_steps,
                "terminated_by": res.trajectory.terminated_by,
                "hop": _hop_label(ins),
                "template_val_attrs": list((ins.meta or {}).get("chain_val_attrs", [])),
                "template_link_attrs": list((ins.meta or {}).get("chain_link_attrs", [])),
            }
        )
        if env.terminated_by == "answer":
            n_answered += 1
        if verbose and (i + 1) % min(20, max(len(insts), 1)) == 0:
            print(
                f"    [{i+1}/{len(insts)}] {tag} acc={st.mean(r['score'] for r in rows):.3f}",
                flush=True,
            )

    accuracy = st.mean(r["score"] for r in rows) if rows else 0.0

    # Stratify by hop
    hop_grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        hop_grouped[r["hop"]].append(r["score"])
    stratified = [
        {"hop": h, "n": len(v), "accuracy": round(st.mean(v), 4)}
        for h, v in sorted(hop_grouped.items())
    ]

    return {
        "tag": tag,
        "n": len(insts),
        "accuracy": round(accuracy, 4),
        "answer_rate": round(n_answered / max(len(insts), 1), 4),
        "mean_n_read": round(st.mean(r["n_read"] for r in rows), 2),
        "mean_n_steps": round(st.mean(r["n_steps"] for r in rows), 2),
        "stratified_by_hop": stratified,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Template holdout diagnostic")
    p.add_argument("--model", default="ckpts/docscout-sft")
    p.add_argument("--train-split", default="data/synth/v4_train1k.json")
    p.add_argument("--eval-split", default="data/synth/v4_eval300.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-eval", type=int, default=0, help="Cap eval instances (0=all)")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--search-k", type=int, default=5)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--adapter", default=None, help="LoRA adapter path")
    p.add_argument("--out", default="results/diagnostic/template_holdout.json")
    p.add_argument(
        "--link-seq",
        nargs="+",
        default=["sync_delay", "rate_limit"],
        help="Link attribute prefix to hold out (default: sync_delay rate_limit — "
        "the most frequent 2-link prefix in eval with 27 train instances).",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip full-eval baseline (useful when the model has already been "
        "evaluated elsewhere and baseline is known).",
    )
    args = p.parse_args()

    train_all = load_split(args.train_split)
    eval_all = load_split(args.eval_split)
    if args.n_eval > 0:
        eval_all = eval_all[: args.n_eval]

    link_seq = tuple(args.link_seq)

    print("=" * 60, flush=True)
    print("Template Holdout Diagnostic", flush=True)
    print(f"  Model     : {args.model}", flush=True)
    print(f"  Train     : {args.train_split}  ({len(train_all)} instances)", flush=True)
    print(f"  Eval      : {args.eval_split}  ({len(eval_all)} instances)", flush=True)
    print(f"  Link seq  : {link_seq}", flush=True)

    # -------------------------------------------------------------------
    # Step 1 — Template distribution (cross-join with eval set)
    # -------------------------------------------------------------------
    print("\n--- Step 1: template distribution ---", flush=True)
    tmpl = analyze_templates(train_all, eval_all)
    print(f"  Train templates: {tmpl['total_train_templates']}", flush=True)
    print(f"  Eval templates:  {tmpl['total_eval_templates']}", flush=True)
    print(f"  Overlap:         {tmpl['overlap_templates']}", flush=True)

    most = tmpl["most_frequent_overlap"]
    least = tmpl["least_frequent_overlap"]

    output: dict = {"template_analysis": tmpl}

    if most:
        most_key = (tuple(most["val_attrs"]), tuple(most["link_attrs"]))
        print(f"  Most-freq overlap: {most_key}  (train={most['train_count']}, eval={most['eval_count']})",
              flush=True)
    else:
        most_key = None
        print("  Most-freq overlap: NONE found — eval and train have disjoint templates", flush=True)

    if least:
        least_key = (tuple(least["val_attrs"]), tuple(least["link_attrs"]))
        print(f"  Least-freq overlap: {least_key}  (train={least['train_count']}, eval={least['eval_count']})",
              flush=True)
    else:
        least_key = None

    # -------------------------------------------------------------------
    # Step 2 — Load model
    # -------------------------------------------------------------------
    print("\n--- Step 2: loading model ---", flush=True)
    client = HFClient(args.model, device=args.device, temperature=args.temp, max_new_tokens=128)
    if args.adapter:
        from peft import PeftModel
        client.model = PeftModel.from_pretrained(client.model, args.adapter).merge_and_unload()
        client.model.eval()
        print(f"  Merged LoRA: {args.adapter}", flush=True)

    env_cfg = EnvConfig(
        max_steps=args.max_steps,
        search_k=args.search_k,
        dynamic_max_steps=False,
    )

    # -------------------------------------------------------------------
    # Step 3 — Baseline: full eval set
    # -------------------------------------------------------------------
    if not args.skip_baseline:
        print("\n--- Step 3: full eval baseline ---", flush=True)
        output["baseline"] = run_eval(client, eval_all, env_cfg, "baseline", verbose=True)
        print(f"  Baseline accuracy: {output['baseline']['accuracy']:.4f}  "
              f"answer_rate={output['baseline']['answer_rate']:.4f}  "
              f"(n={output['baseline']['n']})", flush=True)
    else:
        print("\n--- Step 3: full eval baseline SKIPPED (--skip-baseline) ---", flush=True)

    # -------------------------------------------------------------------
    # Step 4 — Template cross-eval: most-common vs least-common
    # -------------------------------------------------------------------
    print("\n--- Step 4: template cross-eval ---", flush=True)

    baseline_acc = output.get("baseline", {}).get("accuracy", 0.0)

    if most_key:
        eval_most = by_template(eval_all, most_key)
        print(f"  Most-freq eval instances: {len(eval_most)}", flush=True)
        output["most_common_template_eval"] = run_eval(
            client, eval_most, env_cfg, "most_common", verbose=False
        )
        print(f"  Most-common template acc: {output['most_common_template_eval']['accuracy']:.4f}",
              flush=True)
    else:
        output["most_common_template_eval"] = {"note": "no overlapping template found"}

    if least_key:
        eval_least = by_template(eval_all, least_key)
        print(f"  Least-freq eval instances: {len(eval_least)}", flush=True)
        output["least_common_template_eval"] = run_eval(
            client, eval_least, env_cfg, "least_common", verbose=False
        )
        print(f"  Least-common template acc: {output['least_common_template_eval']['accuracy']:.4f}",
              flush=True)
    else:
        output["least_common_template_eval"] = {"note": "no overlapping template found"}

    # -------------------------------------------------------------------
    # Step 5a — Chain-length holdout (4-5 hop vs 2-3 hop)
    # -------------------------------------------------------------------
    print("\n--- Step 5a: chain-length holdout ---", flush=True)

    # Determine which instances belong to which split
    long_eval = [ins for ins in eval_all if (ins.meta or {}).get("K", 0) >= 3]   # 4-5 hop
    short_eval = [ins for ins in eval_all if (ins.meta or {}).get("K", 0) < 3]    # 2-3 hop

    print(f"  Long-chain eval (4-5 hop) : {len(long_eval)}", flush=True)
    print(f"  Short-chain eval (2-3 hop): {len(short_eval)}", flush=True)

    output["chain_length_holdout"] = {
        "description": "Evaluate 4hop+5hop instances to check whether the model "
                       "can handle long chains it was trained on (looks for template "
                       "dependence on chain length).",
        "n_train_total": len(train_all),
        "n_train_short": len(by_chain_length(train_all, max_k=2)),  # 2/3-hop in train
        "n_train_long": len([ins for ins in train_all if (ins.meta or {}).get("K", 0) >= 3]),
    }

    if long_eval:
        output["chain_length_holdout"]["eval_long"] = run_eval(
            client, long_eval, env_cfg, "long_chains", verbose=True
        )
        print(
            f"  Long-chain (4-5 hop) acc : {output['chain_length_holdout']['eval_long']['accuracy']:.4f}",
            flush=True,
        )

    if short_eval:
        output["chain_length_holdout"]["eval_short"] = run_eval(
            client, short_eval, env_cfg, "short_chains", verbose=False
        )
        print(
            f"  Short-chain (2-3 hop) acc: {output['chain_length_holdout']['eval_short']['accuracy']:.4f}",
            flush=True,
        )

    # -------------------------------------------------------------------
    # Step 5b — Link-sequence holdout
    # -------------------------------------------------------------------
    print(f"\n--- Step 5b: link-sequence holdout {link_seq} ---", flush=True)

    # Count how many train instances use this link prefix
    train_with_seq = len(by_link_prefix(train_all, link_seq))
    eval_with_seq = by_link_prefix(eval_all, link_seq)

    print(f"  Train instances with link prefix {link_seq}: {train_with_seq}", flush=True)
    print(f"  Eval  instances with link prefix {link_seq}: {len(eval_with_seq)}", flush=True)

    output["link_sequence_holdout"] = {
        "link_sequence": list(link_seq),
        "n_train_with_prefix": train_with_seq,
        "n_eval_with_prefix": len(eval_with_seq),
    }

    if eval_with_seq:
        output["link_sequence_holdout"]["eval_link_seq"] = run_eval(
            client, eval_with_seq, env_cfg, f"link_seq_{'_'.join(link_seq)}", verbose=True
        )
        print(
            f"  Link-seq holdout acc: {output['link_sequence_holdout']['eval_link_seq']['accuracy']:.4f}",
            flush=True,
        )
    else:
        output["link_sequence_holdout"]["note"] = (
            f"No eval instances have this link prefix. "
            f"Pick a different --link-seq (one that appears in eval)."
        )

    # -------------------------------------------------------------------
    # Step 6 — Synthesis / findings
    # -------------------------------------------------------------------
    print("\n--- Step 6: synthesis ---", flush=True)

    findings: list[str] = []
    baseline_acc = output.get("baseline", {}).get("accuracy", None)
    most_acc = output.get("most_common_template_eval", {}).get("accuracy", None)
    least_acc = output.get("least_common_template_eval", {}).get("accuracy", None)
    long_acc = output.get("chain_length_holdout", {}).get("eval_long", {}).get("accuracy", None)
    short_acc = output.get("chain_length_holdout", {}).get("eval_short", {}).get("accuracy", None)
    link_acc = output.get("link_sequence_holdout", {}).get("eval_link_seq", {}).get("accuracy", None)

    if most_acc is not None and least_acc is not None and most_acc is not None and least_acc is not None:
        gap_tmpl = most_acc - least_acc
        if gap_tmpl > 0.15:
            findings.append(f"TEMPLATE MEMORY DEBT: most-common ({most_acc:.3f}) substantially "
                            f"outperforms least-common template ({least_acc:.3f}, gap={gap_tmpl:+.3f}) — "
                            f"suggests the model is better at frequently-seen templates.")
        elif gap_tmpl > 0.05:
            findings.append(f"Moderate template-frequency effect: most ({most_acc:.3f}) vs "
                            f"least ({least_acc:.3f}), gap={gap_tmpl:+.3f}.")
        else:
            findings.append(f"No strong template-frequency effect: most ({most_acc:.3f}) vs "
                            f"least ({least_acc:.3f}), gap={gap_tmpl:+.3f}.")

    if long_acc is not None and baseline_acc is not None:
        gap_long = baseline_acc - long_acc
        if abs(gap_long) < 0.05:
            findings.append(f"Long-chain (4-5 hop) accuracy ({long_acc:.3f}) close to baseline "
                            f"({baseline_acc:.3f}) — model generalizes across chain lengths.")
        elif gap_long > 0.10:
            findings.append(f"PERFORMANCE GAP on long chains: {long_acc:.3f} vs baseline {baseline_acc:.3f} "
                            f"(gap={gap_long:+.3f}) — model may rely on chain-length-specific patterns.")
        else:
            findings.append(f"Modest gap on long chains: {long_acc:.3f} vs baseline {baseline_acc:.3f} "
                            f"(gap={gap_long:+.3f}).")

    if link_acc is not None and baseline_acc is not None:
        gap_link = baseline_acc - link_acc
        if abs(gap_link) < 0.05:
            findings.append(f"Link-sequence holdout accuracy ({link_acc:.3f}) matches baseline "
                            f"({baseline_acc:.3f}) — no evidence of link-sequence memorization.")
        elif gap_link > 0.10:
            findings.append(f"LINK-SEQUENCE MEMORY DEBT: holdout accuracy {link_acc:.3f} vs "
                            f"baseline {baseline_acc:.3f} (gap={gap_link:+.3f}). "
                            f"Model may depend on training exposure to specific attribute sequences.")
        else:
            findings.append(f"Modest link-sequence effect: {link_acc:.3f} vs baseline "
                            f"{baseline_acc:.3f} (gap={gap_link:+.3f}).")

    # Template memorisation verdict
    max_gap = 0.0
    if most_acc is not None and least_acc is not None:
        max_gap = max(max_gap, abs(most_acc - least_acc))
    if long_acc is not None and baseline_acc is not None:
        max_gap = max(max_gap, abs(long_acc - baseline_acc))
    if link_acc is not None and baseline_acc is not None:
        max_gap = max(max_gap, abs(link_acc - baseline_acc))

    verdict = (
        "CLEAN — no evidence of template memorisation; model appears to use "
        "general search strategy across templates."
        if max_gap < 0.08
        else "MIXED — moderate template dependence on some axes."
        if max_gap < 0.15
        else "SUSPICIOUS — large accuracy gaps across holdout axes suggest "
        "the model may be memorising template patterns rather than learning "
        "a general multi-step search policy."
    )
    findings.append(f"Verdict: {verdict}")

    output["analysis"] = {
        "verdict": verdict,
        "max_holdout_gap": round(max_gap, 4),
        "findings": findings,
    }

    print(f"\n  === VERDICT ===", flush=True)
    for f in findings:
        print(f"  - {f}", flush=True)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)

    return output


if __name__ == "__main__":
    main()
