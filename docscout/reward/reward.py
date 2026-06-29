"""Reward engine for DocScout — the core novelty carrier.

Three reward variants (EXPERIMENT_PLAN.md claim C2 / FINAL_PROPOSAL.md):
  - R_add    : answer + evidence − λ·read_tokens              (additive cost)
  - R_redund : answer + evidence − β·redundancy − λ·read_tokens (ALDEN-flavored)
  - R_ratio  : answer + evidence − γ·(1 − efficiency_ratio)   (ours, anti-hack)

`efficiency_ratio = gold_tokens_read / total_tokens_read` attributes how much of
what the agent *read* was actually gold evidence — the anti-"read everything"
term. Evidence is computed from *sections read* (robust to brittle citation
parsing) by default; switchable to cited evidence via config.

Uses a `@reward` registry mirroring CodeScout's reward interface so the same
function plugs into a SkyRL-style training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from docscout.env.search_env import SearchEnv
from docscout.reward.answer_scoring import score_answer
from docscout.types import QAInstance

REWARD_REGISTRY: dict[str, Callable] = {}


def reward(name: str):
    def deco(fn):
        REWARD_REGISTRY[name] = fn
        return fn
    return deco


@dataclass
class RewardConfig:
    w_answer: float = 1.0
    w_evidence: float = 0.3
    evidence_source: str = "read"  # "read" (sections read) | "cited"
    # additive cost
    lambda_tokens: float = 5e-4
    # redundancy (ALDEN-flavored)
    beta_redundancy: float = 5e-2
    # efficiency ratio (ours)
    gamma_ratio: float = 1.0
    # per-action cost (Round 10): penalizes wasteful read/expand ACTIONS (e.g.
    # re-reading already-seen sections) which the committed-TOKEN cost misses.
    # Default 0 preserves existing ratio/add/redund semantics.
    action_cost: float = 0.0
    # SELECTION bonus (Round 11): rewards the agent for reading GOLD evidence,
    # especially early — directly targets the selection bottleneck (the read-budget
    # token/efficiency terms cannot shape WHICH candidate is read).
    w_gold_hit: float = 0.0       # +w if ANY committed read is gold
    w_first_gold: float = 0.0     # +w if the FIRST committed read is gold
    # termination shaping (prevents "exhaust budget without submitting", cf. CodeScout r_turn)
    r_submit_bonus: float = 0.5
    r_nosubmit_penalty: float = -0.5


# --------------------------------------------------------------------------- helpers
def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evidence_f1_read(env: SearchEnv, gold: set[tuple[str, str]]) -> float:
    """Section-F1 over *committed reads* (READ/EXPAND) vs gold.

    Snippet glimpses don't count — the agent must *commit* to reading a section
    for it to count as verified evidence. This is the read-budget evidence signal.
    """
    read = env.committed_read_uids()
    if not read or not gold:
        return 0.0
    tp = len(read & gold)
    precision = tp / len(read)
    recall = tp / len(gold)
    return _f1(precision, recall)


def evidence_f1_cited(env: SearchEnv, gold: set[tuple[str, str]]) -> float:
    """Section-F1 over *cited* evidence vs gold."""
    cited = {tuple(e) for e in env.final_evidence}
    if not cited or not gold:
        return 0.0
    tp = len(cited & gold)
    return _f1(tp / len(cited), tp / len(gold))


def redundancy_count(env: SearchEnv) -> int:
    """Duplicate reads (re-reading a section) + repeated identical searches."""
    seen: set[tuple[str, str]] = set()
    dup_reads = 0
    seen_q: set[str] = set()
    dup_search = 0
    for e in env.read_log:
        k = (e.doc_id, e.section_id)
        if k in seen:
            dup_reads += 1
        seen.add(k)
    for a in env.action_log:
        if a.startswith("search("):
            q = a
            if q in seen_q:
                dup_search += 1
            seen_q.add(q)
    return dup_reads + dup_search


def _components(env: SearchEnv, inst: QAInstance, cfg: RewardConfig) -> dict:
    gold = inst.gold_sections()
    answer = score_answer(env.final_answer, inst.gold_answer) if env.final_answer else 0.0
    if cfg.evidence_source == "cited":
        ev = evidence_f1_cited(env, gold)
    else:
        ev = evidence_f1_read(env, gold)
    read_tokens = env.total_read_tokens
    gold_tokens = env.gold_tokens_read()  # anti-pollution: gold tokens in context (snippet+committed)
    committed_tok = env.committed_read_tokens
    committed_gold = env.gold_tokens_read(committed_only=True)
    # efficiency_ratio over COMMITTED reads only: an efficient agent that searches
    # once then reads ONLY gold gets ratio -> 1.0 (not taxed by non-gold snippets).
    eff_ratio = (committed_gold / committed_tok) if committed_tok > 0 else 0.0
    redundancy = redundancy_count(env)
    term = cfg.r_submit_bonus if env.terminated_by == "answer" else cfg.r_nosubmit_penalty
    n_actions = env.n_read + env.n_expand  # committed read ACTIONS (incl. wasteful re-reads)
    # SELECTION signals (Round 11): did the agent read gold, and did it read gold FIRST?
    committed = [e for e in env.read_log if e.source != "search_snippet"]
    gold_hit = 1.0 if any((e.doc_id, e.section_id) in gold for e in committed) else 0.0
    first_gold = 1.0 if (committed and (committed[0].doc_id, committed[0].section_id) in gold) else 0.0
    return dict(
        answer=answer, evidence=ev, read_tokens=read_tokens, gold_tokens_read=gold_tokens,
        committed_read_tokens=committed_tok, committed_gold_tokens=committed_gold,
        efficiency_ratio=eff_ratio, redundancy=redundancy, termination=term,
        n_actions=n_actions, action_cost=cfg.action_cost,
        gold_hit=gold_hit, first_gold=first_gold,
        terminated_by=env.terminated_by, n_steps=env.n_steps,
    )


# --------------------------------------------------------------------------- variants
def _sel_bonus(cfg, c):
    return cfg.w_gold_hit * c["gold_hit"] + cfg.w_first_gold * c["first_gold"]


@reward("additive")
def r_additive(env: SearchEnv, inst: QAInstance, cfg: RewardConfig) -> tuple[float, dict]:
    c = _components(env, inst, cfg)
    r = cfg.w_answer * c["answer"] + cfg.w_evidence * c["evidence"] \
        - cfg.lambda_tokens * c["read_tokens"] - cfg.action_cost * c["n_actions"] \
        + _sel_bonus(cfg, c) + c["termination"]
    return r, c


@reward("redundancy")
def r_redundancy(env: SearchEnv, inst: QAInstance, cfg: RewardConfig) -> tuple[float, dict]:
    c = _components(env, inst, cfg)
    r = cfg.w_answer * c["answer"] + cfg.w_evidence * c["evidence"] \
        - cfg.beta_redundancy * c["redundancy"] - cfg.lambda_tokens * c["read_tokens"] \
        - cfg.action_cost * c["n_actions"] + _sel_bonus(cfg, c) + c["termination"]
    return r, c


@reward("ratio")
def r_ratio(env: SearchEnv, inst: QAInstance, cfg: RewardConfig) -> tuple[float, dict]:
    c = _components(env, inst, cfg)
    # efficiency-ratio cost only bites when the agent actually COMMITTED a read
    ratio_cost = cfg.gamma_ratio * (1.0 - c["efficiency_ratio"]) if c["committed_read_tokens"] > 0 else 0.0
    r = (cfg.w_answer * c["answer"] + cfg.w_evidence * c["evidence"] - ratio_cost
         - cfg.action_cost * c["n_actions"] + _sel_bonus(cfg, c) + c["termination"])
    return r, c


def compute_reward(name: str, env: SearchEnv, inst: QAInstance,
                   cfg: RewardConfig | None = None) -> tuple[float, dict]:
    """Dispatch by registered name. Returns (scalar_reward, components_dict)."""
    if cfg is None:
        cfg = RewardConfig()
    if name not in REWARD_REGISTRY:
        raise KeyError(f"unknown reward '{name}'; available: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name](env, inst, cfg)
