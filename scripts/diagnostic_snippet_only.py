"""Diagnostic: snippet-only accuracy vs normal accuracy.

Purpose: Measure how well an agent can answer when it is BLOCKED from reading
full sections — it can only use search snippets (40-word BM25 excerpts) + answer.

If snippet-only accuracy is high (>0.6), the BM25 snippets are leaking the
answer, meaning the environment is too easy and the read-budget reward has
nothing to learn. If it's near-zero, snippets are clean and reading adds real
value.

Also runs a one-shot RAG baseline (top-5 snippets -> model QA) for comparison.

Usage:
  python -m scripts.diagnostic_snippet_only \
      --model ckpts/docscout-sft \
      --splits data/synth/v3_eval300.json data/synth/v4_eval300.json \
      -n 100 --device cuda:0 \
      --out results/diagnostic/snippet_only.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from docscout.agent.client import HFClient
from docscout.agent.rollout import (
    SYSTEM_PROMPT,
    _recent_reads,
    build_messages,
    rollout,
)
from docscout.env.docstore import DocStore
from docscout.env.search_env import EnvConfig, SearchEnv
from docscout.reward.answer_scoring import score_answer
from docscout.types import Document, EvidenceSpan, QAInstance, Section
from docscout.agent.parsing import parse_action
from docscout.types import Trajectory


# ---------------------------------------------------------------------------
# Data loading (reused from run_retriever_eval.py)
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


# ---------------------------------------------------------------------------
# Snippet-only env: blocks read/expand actions
# ---------------------------------------------------------------------------

SNIPPET_BLOCKED_MSG = "BLOCKED: read not available — you may only use search (snippets) and answer."


class SnippetOnlySearchEnv(SearchEnv):
    """Drop-in replacement for SearchEnv that blocks READ and EXPAND actions.

    The agent can still SEARCH (gets snippets) and ANSWER.  Any attempt to
    read or expand returns a blocked message with zero token cost.  The action
    is still logged so trajectory analysis is possible.
    """

    def step(self, action):
        from docscout.types import ActionType, StepResult

        if self.done:
            return StepResult(observation="(episode already ended)", done=True)

        self.n_steps += 1

        if action.type == ActionType.SEARCH:
            obs, rt = self._do_search(action)
        elif action.type == ActionType.READ:
            self.n_read += 1
            query = str(action.args.get("doc_id", "") or action.args.get("idx", ""))
            self.action_log.append(f"read({query}) -> BLOCKED (snippets only)")
            obs = SNIPPET_BLOCKED_MSG
            rt = 0
        elif action.type == ActionType.EXPAND:
            self.n_expand += 1
            q = str(action.args.get("doc_id", ""))
            d = str(action.args.get("direction", ""))
            self.action_log.append(f"expand({q},{d}) -> BLOCKED (snippets only)")
            obs = SNIPPET_BLOCKED_MSG
            rt = 0
        elif action.type == ActionType.ANSWER:
            return self._do_answer(action)
        else:
            obs, rt = f"(unknown action: {action.type})", 0

        self.total_read_tokens += rt
        over_budget = False

        if self.n_steps >= self.cfg.max_steps:
            self.done = True
            self.terminated_by = "step_budget"
            obs += "\n[step budget exhausted — you must answer now]"
            over_budget = True
        if self.total_read_tokens >= self.cfg.max_read_tokens:
            self.done = True
            if self.terminated_by == "":
                self.terminated_by = "token_budget"
            obs += "\n[read-token budget exhausted — you must answer now]"
            over_budget = True

        return StepResult(observation=obs, done=self.done, read_tokens=rt, info={"over_budget": over_budget})


def build_snippet_messages(env: SearchEnv, recent_reads_kept: int | None = None) -> list[dict]:
    """Build messages with a modified system prompt that says read/expand are unavailable."""
    snippet_system = (
        "You are a document-search agent. You must locate the answer to a question "
        "in a corpus using ONLY these tools:\n\n"
        "- search  : args {query}            -> returns top-k section SNIPPETS\n"
        "- answer  : args {text, evidence}   -> submit final answer & end\n\n"
        "IMPORTANT: You do NOT have access to read or expand tools. "
        "You can only use search snippets to answer the question.\n\n"
        "Reply with EXACTLY one action block:\n"
        "ACTION: <search|answer>\n"
        "<field>: <value>\n"
        "..."
    )

    cands = "\n".join(
        f"  {i+1}. [{h['doc_id']}.{h['section_id']}] {h['section_title']} — {h['snippet']}"
        for i, h in enumerate(env.current_candidates)
    ) or "  (no search yet)"
    actions = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(env.action_log[-12:])) or "  (none)"
    user = (
        f"QUESTION: {env.question}\n\n"
        f"ACTIONS SO FAR:\n{actions}\n\n"
        f"CURRENT SNIPPET CANDIDATES:\n{cands}\n\n"
        f"Read budget left: steps {env.cfg.max_steps - env.n_steps}, "
        f"tokens {max(0, env.cfg.max_read_tokens - env.total_read_tokens)}.\n"
        f"Decide the next action (search or answer only)."
    )
    return [{"role": "system", "content": snippet_system}, {"role": "user", "content": user}]


def force_snippet_answer_messages(env: SearchEnv) -> list[dict]:
    """Force-answer prompt for snippet-only mode."""
    snippets = "\n".join(
        f"[{h['doc_id']}.{h['section_id']}] {h['section_title']} — {h['snippet']}"
        for h in env.current_candidates
    ) or "(no snippets collected)"

    history = "\n".join(env.action_log) or "(none)"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"QUESTION: {env.question}\n\n"
            f"You have run out of steps. Using ONLY the snippets you collected, "
            f"you MUST submit your final answer NOW.\n\n"
            f"SEARCH HISTORY:\n{history}\n\n"
            f"LAST SNIPPETS:\n{snippets}\n\n"
            f"Reply with exactly:\nACTION: answer\ntext: <your answer>"
        )},
    ]


def snippet_only_rollout(
    instance: QAInstance,
    client: HFClient,
    env_config: EnvConfig | None = None,
    max_steps_override: int | None = None,
) -> tuple[dict, SearchEnv]:
    """Run one episode where the agent can only use search + answer.

    Returns (row_dict, env).
    """
    cfg = env_config or EnvConfig()
    if max_steps_override is not None:
        cfg = EnvConfig(
            max_steps=max_steps_override,
            max_read_tokens=cfg.max_read_tokens,
            search_k=cfg.search_k,
            snippet_token_cost=cfg.snippet_token_cost,
            rerank=cfg.rerank,
        )

    effective_max_steps = cfg.effective_max_steps(instance) if max_steps_override is None else max_steps_override
    env_cfg = EnvConfig(
        max_steps=effective_max_steps,
        max_read_tokens=cfg.max_read_tokens,
        search_k=cfg.search_k,
        snippet_token_cost=cfg.snippet_token_cost,
        rerank=cfg.rerank,
        dynamic_max_steps=False,
    )

    store = DocStore(instance.docs, rerank=env_cfg.rerank)
    env = SnippetOnlySearchEnv(instance, store, env_cfg)
    traj = Trajectory(instance_id=instance.instance_id)

    while not env.done:
        if env.n_steps >= effective_max_steps - 1:
            messages = force_snippet_answer_messages(env)
        else:
            messages = build_snippet_messages(env)

        raw = client.act(env, messages)
        action = parse_action(raw)
        traj.actions.append(action)
        step = env.step(action)
        traj.observations.append(step.observation)

    traj.final_answer = env.final_answer
    traj.final_evidence = [(d, s) for (d, s) in env.final_evidence]
    traj.total_read_tokens = env.total_read_tokens
    traj.n_search, traj.n_read, traj.n_expand, traj.n_steps = (
        env.n_search, env.n_read, env.n_expand, env.n_steps
    )
    traj.terminated_by = env.terminated_by

    sc = score_answer(traj.final_answer, str(instance.gold_answer))
    row = {
        "instance_id": instance.instance_id,
        "gold": str(instance.gold_answer),
        "pred": traj.final_answer[:80],
        "score": sc,
        "n_search": traj.n_search,
        "n_read_blocked": traj.n_read,
        "n_expand_blocked": traj.n_expand,
        "n_steps": traj.n_steps,
        "terminated": traj.terminated_by,
        "total_snippet_tokens": traj.total_read_tokens,
    }
    return row, env


# ---------------------------------------------------------------------------
# Normal rollout wrapper
# ---------------------------------------------------------------------------

def normal_rollout_row(
    instance: QAInstance,
    client: HFClient,
    env_config: EnvConfig | None = None,
    max_steps_override: int | None = None,
) -> tuple[dict, SearchEnv]:
    """Run a standard rollout with full tool access. Returns (row_dict, env)."""
    from docscout.agent.rollout import rollout as _rollout

    res, env = _rollout(
        instance,
        client,
        env_config=env_config,
        max_steps_override=max_steps_override,
        return_env=True,
    )
    sc = score_answer(res.trajectory.final_answer, str(instance.gold_answer))
    row = {
        "instance_id": instance.instance_id,
        "gold": str(instance.gold_answer),
        "pred": res.trajectory.final_answer[:80],
        "score": sc,
        "n_search": res.trajectory.n_search,
        "n_read": res.trajectory.n_read,
        "n_expand": res.trajectory.n_expand,
        "n_steps": res.trajectory.n_steps,
        "terminated": res.trajectory.terminated_by,
        "total_read_tokens": res.trajectory.total_read_tokens,
    }
    return row, env


# ---------------------------------------------------------------------------
# One-shot RAG baseline (top-5 snippets -> model QA)
# ---------------------------------------------------------------------------

ONESHOT_RAG_PROMPT = (
    "You are given a question and up to 5 document snippets (short excerpts).\n"
    "Answer the question using ONLY these snippets.\n"
    "If the snippets do not contain enough information, give your best guess.\n"
    "Reply with ONLY the answer — a short phrase, number, or entity name.\n\n"
    "Snippets:\n{snippets}\n\n"
    "Question: {question}\nAnswer:"
)


def run_oneshot_rag_snippet(
    client: HFClient,
    instance: QAInstance,
    k: int = 5,
    max_tokens: int = 48,
) -> dict:
    """Search with question, feed top-k snippets to model, score."""
    store = DocStore(instance.docs)
    hits = store.search(instance.question, k=k)
    snippets = "\n".join(
        f"[{i+1}] [{h['doc_id']}.{h['section_id']}] {h['section_title']}: {h['snippet']}"
        for i, h in enumerate(hits)
    )
    prompt = ONESHOT_RAG_PROMPT.format(snippets=snippets, question=instance.question)
    pred = client.complete(prompt, max_tokens=max_tokens, temperature=0.0)
    sc = score_answer(pred, str(instance.gold_answer))
    return {
        "instance_id": instance.instance_id,
        "gold": str(instance.gold_answer),
        "pred": pred[:80],
        "score": sc,
        "n_snippets": len(hits),
    }


# ---------------------------------------------------------------------------
# Hop-stratified accuracy helper
# ---------------------------------------------------------------------------

def get_hop_label(inst: QAInstance) -> str:
    """Return a human-readable hop label for the instance."""
    meta = inst.meta or {}
    kind = meta.get("kind", "")
    if kind == "single_hop":
        return "1-hop"
    elif kind == "multi_hop":
        return "2-hop"
    elif "chain" in kind:
        K = meta.get("K", 0)
        return f"{K+1}-hop"
    return "unknown"


def stratify_accuracy(rows: list[dict], insts: list[QAInstance], key: str = "score") -> list[dict]:
    """Compute accuracy per hop group."""
    label_map = {i: get_hop_label(insts[i]) for i in range(len(insts))}
    groups: dict[str, list[float]] = {}
    for i, row in enumerate(rows):
        lbl = label_map.get(i, "unknown")
        groups.setdefault(lbl, []).append(row[key])
    results = []
    for lbl in sorted(groups):
        vals = groups[lbl]
        results.append({
            "group": lbl,
            "n": len(vals),
            "accuracy": round(st.mean(vals), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ckpts/docscout-sft")
    p.add_argument("--splits", nargs="+", required=True, help="Paths to JSON split files")
    p.add_argument("-n", type=int, default=None, help="Limit instances per split")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--search-k", type=int, default=5)
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", default="results/diagnostic/snippet_only.json")
    p.add_argument("--skip-normal", action="store_true", help="Skip normal agent eval (already have results)")
    p.add_argument("--skip-rag", action="store_true", help="Skip one-shot RAG baseline")
    args = p.parse_args()

    client = HFClient(
        args.model, device=args.device, temperature=args.temp, max_new_tokens=96
    )
    if args.adapter:
        from peft import PeftModel
        client.model = PeftModel.from_pretrained(client.model, args.adapter).merge_and_unload()
        client.model.eval()
        print(f"[eval] merged LoRA adapter {args.adapter}", flush=True)

    env_cfg = EnvConfig(
        max_steps=args.max_steps, search_k=args.search_k, dynamic_max_steps=False
    )

    output = {"model": args.model, "device": args.device, "tag": "snippet_only_diagnostic", "splits": {}}

    for split_path in args.splits:
        split_name = Path(split_path).stem
        print(f"\n{'='*60}", flush=True)
        print(f"Processing {split_name}...", flush=True)
        insts = load_split(split_path)
        if args.n:
            insts = insts[: args.n]
        print(f"  {len(insts)} instances", flush=True)

        split_output = {"n": len(insts)}

        # 1. Snippet-only agent
        print(f"  [1/3] Running snippet-only agent eval...", flush=True)
        snippet_rows = []
        n_ans_snippet = 0
        for i, inst in enumerate(insts):
            row, env = snippet_only_rollout(inst, client, env_config=env_cfg)
            snippet_rows.append(row)
            if env.terminated_by == "answer":
                n_ans_snippet += 1
            if (i + 1) % 20 == 0:
                acc = st.mean(r["score"] for r in snippet_rows)
                print(f"    [{i+1}/{len(insts)}] snippet-only acc={acc:.3f}", flush=True)

        snippet_acc = st.mean(r["score"] for r in snippet_rows) if snippet_rows else 0.0
        snippet_stratified = stratify_accuracy(snippet_rows, insts)
        split_output["snippet_only"] = {
            "accuracy": round(snippet_acc, 4),
            "answer_rate": round(n_ans_snippet / len(insts), 4) if insts else 0,
            "mean_n_search": round(st.mean(r["n_search"] for r in snippet_rows), 2),
            "mean_n_steps": round(st.mean(r["n_steps"] for r in snippet_rows), 2),
            "stratified_by_hop": snippet_stratified,
        }
        print(f"  snippet_only acc={snippet_acc:.4f}", flush=True)

        # 2. Normal agent (full tool access)
        if not args.skip_normal:
            print(f"  [2/3] Running normal agent eval...", flush=True)
            normal_rows = []
            n_ans_normal = 0
            for i, inst in enumerate(insts):
                row, env = normal_rollout_row(inst, client, env_config=env_cfg)
                normal_rows.append(row)
                if env.terminated_by == "answer":
                    n_ans_normal += 1
                if (i + 1) % 20 == 0:
                    acc = st.mean(r["score"] for r in normal_rows)
                    print(f"    [{i+1}/{len(insts)}] normal acc={acc:.3f}", flush=True)

            normal_acc = st.mean(r["score"] for r in normal_rows) if normal_rows else 0.0
            normal_stratified = stratify_accuracy(normal_rows, insts)
            split_output["normal_agent"] = {
                "accuracy": round(normal_acc, 4),
                "answer_rate": round(n_ans_normal / len(insts), 4) if insts else 0,
                "mean_n_read": round(st.mean(r.get("n_read", 0) for r in normal_rows), 2),
                "mean_n_steps": round(st.mean(r["n_steps"] for r in normal_rows), 2),
                "stratified_by_hop": normal_stratified,
            }
            print(f"  normal_agent acc={normal_acc:.4f}", flush=True)
        else:
            print(f"  [2/3] Skipping normal agent eval (--skip-normal set)", flush=True)

        # 3. One-shot RAG baseline
        if not args.skip_rag:
            print(f"  [3/3] Running one-shot RAG baseline...", flush=True)
            rag_rows = []
            for i, inst in enumerate(insts):
                row = run_oneshot_rag_snippet(client, inst, k=5)
                rag_rows.append(row)
                if (i + 1) % 20 == 0:
                    acc = st.mean(r["score"] for r in rag_rows)
                    print(f"    [{i+1}/{len(insts)}] RAG acc={acc:.3f}", flush=True)

            rag_acc = st.mean(r["score"] for r in rag_rows) if rag_rows else 0.0
            rag_stratified = stratify_accuracy(rag_rows, insts)
            split_output["oneshot_rag_snippet"] = {
                "accuracy": round(rag_acc, 4),
                "stratified_by_hop": rag_stratified,
            }
            print(f"  oneshot_rag acc={rag_acc:.4f}", flush=True)
        else:
            print(f"  [3/3] Skipping RAG baseline (--skip-rag set)", flush=True)

        # Gap analysis
        if "normal_agent" in split_output:
            gap = split_output["normal_agent"]["accuracy"] - split_output["snippet_only"]["accuracy"]
            split_output["gap_normal_vs_snippet"] = round(gap, 4)

        # Verdict
        split_output["verdict"] = (
            "SNIPPET_LEAKAGE" if snippet_acc > 0.6
            else "SNIPPET_INSUFFICIENT" if snippet_acc < 0.3
            else "MIXED"
        )

        # Detailed rows (snippet-only only; normal + RAG on request)
        split_output["snippet_only_rows"] = snippet_rows
        if not args.skip_normal:
            split_output["normal_rows"] = normal_rows
        if not args.skip_rag:
            split_output["rag_rows"] = rag_rows

        output["splits"][split_name] = split_output

        # Print summary
        print(f"\n  === {split_name} SUMMARY ===")
        print(f"  Snippet-only accuracy: {snippet_acc:.4f}")
        if not args.skip_normal:
            print(f"  Normal agent accuracy: {normal_acc:.4f}")
            print(f"  Gap: {gap:.4f}")
        if not args.skip_rag:
            print(f"  One-shot RAG accuracy: {rag_acc:.4f}")
        print(f"  Verdict: {split_output['verdict']}")

        for s in snippet_stratified:
            print(f"    Hop {s['group']}: n={s['n']} acc={s['accuracy']:.4f}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)

    return output


if __name__ == "__main__":
    main()