"""Failure classification harness (自动化实验迭代方案.md §四).

Every failed trajectory is attributed to ONE dominant cause so iteration targets
the right bottleneck (retrieval->index/data, selection->obs/SFT, stopping->cost,
answer->format/model), not blindly "more RL".

Takes a finished rollout (env + instance) and returns a label. Priority order:
  reward_hack > stop_fail > reading_fail > selection_fail > retrieval_fail > answer_fail > correct
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from docscout.env.search_env import SearchEnv
from docscout.reward.answer_scoring import score_answer
from docscout.types import QAInstance


@dataclass
class FailureBreakdown:
    distribution: dict[str, float]
    counts: dict[str, int]
    examples: dict[str, list[str]]
    n: int


def classify(env: SearchEnv, inst: QAInstance) -> str:
    gold = inst.gold_sections()
    strict_ok = score_answer(env.final_answer, inst.gold_answer) == 1.0   # strict EM for routing
    partial = (not strict_ok) and score_answer(env.final_answer, inst.gold_answer) > 0
    all_cands_uids = {  # every section ever surfaced (snippet or read)
        (e.doc_id, e.section_id) for e in env.read_log}
    committed = env.committed_read_uids()
    gold_committed = committed & gold
    gold_surfaced = all_cands_uids & gold
    hack_thresh = max(4, env.cfg.max_steps - 2)  # adaptive to budget

    if strict_ok:
        # correct but committed far too many reads -> likely covering to hit gold
        if len(committed) >= hack_thresh:
            return "reward_hack"
        return "correct"
    if partial:
        # close but not exact -> reading/generation (not a clean correct)
        return "reading_fail"
    # wrong/no answer -> why?
    if env.terminated_by != "answer":
        return "stop_fail"                      # never submitted -> ran out of budget
    if env.n_steps <= 2 and not gold_committed:
        return "stop_fail"                      # answered after almost no search/read
    if gold_committed:
        return "reading_fail"                   # had gold in context but answered wrong
    if gold_surfaced:
        return "selection_fail"                 # gold in a snippet but never committed-read
    if env.n_search == 0:
        return "stop_fail"
    return "retrieval_fail"                     # gold never surfaced at all


def analyze(records: list[tuple[SearchEnv, QAInstance]], k_examples: int = 3) -> FailureBreakdown:
    """records = list of (env, instance) after rollout. Returns distribution + examples."""
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for env, inst in records:
        label = classify(env, inst)
        counts[label] += 1
        if label != "correct" and len(examples.get(label, [])) < k_examples:
            examples.setdefault(label, []).append(
                f"{inst.instance_id}: Q='{inst.question[:50]}…' gold={inst.gold_answer} "
                f"pred='{env.final_answer[:30]}' steps={env.n_steps} reads={len(env.committed_read_uids())}")
    n = sum(counts.values()) or 1
    dist = {k: counts[k] / n for k in counts}
    return FailureBreakdown(distribution=dist, counts=dict(counts), examples=examples, n=sum(counts.values()))


def main():
    import argparse, json
    p = argparse.ArgumentParser(description="Failure analysis is invoked from run_rollout/eval scripts; "
                                            "this main() runs it on saved trajectories.")
    p.add_argument("--trajectories", required=True, help="JSON of saved rollout env states")
    args = p.parse_args()
    print("(use analyze() from eval scripts; see scripts/run_eval.py for wiring)")


if __name__ == "__main__":
    main()
