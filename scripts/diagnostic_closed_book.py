"""Closed-book audit: measure pre-trained knowledge baseline.

Given a model and one or more QA splits, asks the model to answer each question
WITHOUT any provided documents, then scores with the same `score_answer` used in
the agent loop.  A near-zero accuracy on synth data confirms the questions are not
trivially memorizable; non-zero accuracy on grounded data (MuSiQue) signals
pre-training contamination.

Usage:
  python -m scripts.diagnostic_closed_book \
      --model ckpts/docscout-sft \
      --splits data/synth/v4_eval300.json data/grounded/musique_test.json \
      -n 300 --device cuda:0 --out results/diagnostic/closed_book.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from docscout.agent.client import HFClient
from docscout.reward.answer_scoring import score_answer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(path: str) -> list[dict]:
    """Load a JSON QA split into a flat list of dicts."""
    raw = json.load(open(path))
    out = []
    for r in raw:
        out.append({
            "instance_id": r["instance_id"],
            "question": r["question"],
            "gold_answer": str(r["gold_answer"]),
            "meta": r.get("meta", {}),
            "question_len": len(r["question"].split()),
        })
    return out


# ---------------------------------------------------------------------------
# Closed-book prompt
# ---------------------------------------------------------------------------

CB_PROMPT = (
    "Answer the following question based on your own knowledge. "
    "Reply with ONLY the answer — a short phrase or number. Do not include reasoning.\n\n"
    "Question: {question}\nAnswer:"
)


def run_closed_book(client: HFClient, insts: list[dict],
                    max_tokens: int = 48, temperature: float = 0.0) -> list[dict]:
    """Ask each question without documents; return per-instance rows."""
    rows = []
    for i, inst in enumerate(insts):
        prompt = CB_PROMPT.format(question=inst["question"])
        pred = client.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        sc = score_answer(pred, inst["gold_answer"])
        rows.append({
            "instance_id": inst["instance_id"],
            "question": inst["question"],
            "gold_answer": inst["gold_answer"],
            "pred": pred,
            "score": sc,
            "question_len": inst["question_len"],
        })
        if (i + 1) % 30 == 0:
            acc = st.mean(r["score"] for r in rows)
            print(f"  [{i+1}/{len(insts)}] running closed-book acc={acc:.3f}", flush=True)
    return rows


def bucket_accuracy(rows: list[dict], bins: tuple = (10, 15, 20, 25, 30, 35, 40)):
    """Accuracy by question-length buckets (word count)."""
    results = []
    bounds = [0] + list(bins) + [9999]
    for lo, hi in zip(bounds, bounds[1:]):
        label = f"{lo}-{hi}" if hi < 9999 else f"{lo}+"
        subset = [r for r in rows if lo <= r["question_len"] < hi]
        if not subset:
            continue
        results.append({
            "bucket": label,
            "n": len(subset),
            "acc": round(st.mean(r["score"] for r in subset), 4),
            "sum_score": round(sum(r["score"] for r in subset), 2),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ckpts/docscout-sft")
    p.add_argument("--splits", nargs="+", required=True,
                   help="Paths to JSON split files")
    p.add_argument("--split-names", nargs="+", default=None,
                   help="Labels for each split (default: filename stem)")
    p.add_argument("-n", type=int, default=None,
                   help="Limit instances per split (None = all)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default="results/diagnostic/closed_book.json")
    args = p.parse_args()

    client = HFClient(args.model, device=args.device, temperature=args.temperature,
                      max_new_tokens=args.max_tokens)

    all_summaries = {}
    all_rows = {}

    for idx, split_path in enumerate(args.splits):
        name = args.split_names[idx] if args.split_names else Path(split_path).stem
        print(f"\n=== Closed-book: {name} ===", flush=True)
        insts = load_split(split_path)
        total_available = len(insts)
        if args.n:
            insts = insts[: args.n]
        print(f"  {len(insts)} instances (of {total_available} available)", flush=True)

        rows = run_closed_book(client, insts, max_tokens=args.max_tokens,
                               temperature=args.temperature)
        acc = st.mean(r["score"] for r in rows) if rows else 0.0
        bucketed = bucket_accuracy(rows)

        # correctly answered (any credit > 0)
        correct = [r for r in rows if r["score"] > 0]
        full_correct = [r for r in rows if r["score"] >= 1.0]
        partial_correct = [r for r in rows if 0 < r["score"] < 1.0]

        summary = {
            "split": name,
            "split_path": split_path,
            "n": len(rows),
            "accuracy": round(acc, 4),
            "num_correct_any": len(correct),
            "num_full_correct": len(full_correct),
            "num_partial_correct": len(partial_correct),
            "bucket_accuracy": bucketed,
            "correct_instances": [{
                "instance_id": r["instance_id"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "pred": r["pred"],
                "score": r["score"],
                "question_len": r["question_len"],
            } for r in correct],
        }
        all_summaries[name] = summary
        all_rows[name] = rows

        print(f"  accuracy={acc:.4f}  correct_any={len(correct)}/{len(rows)}  "
              f"full={len(full_correct)} partial={len(partial_correct)}", flush=True)
        for b in bucketed:
            print(f"    {b['bucket']:>6s}: n={b['n']:3d} acc={b['acc']:.4f}", flush=True)

    # Assemble output (rows stored by split name; correct_instances already inside summaries)
    output = {
        "model": args.model,
        "device": args.device,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "summaries": all_summaries,
        "rows": {name: all_rows[name][:args.n] if args.n else all_rows[name] for name in all_rows},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
