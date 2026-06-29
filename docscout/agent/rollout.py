"""Rollout loop: build the three-tier-memory prompt, call the client, step the env.

The prompt re-renders a *compact* state each turn (question + action_log +
current candidates + last-N read sections) rather than dumping every observed
token — this is the context-control mechanism the contribution is built on
(RESEARCH_BRIEF.md §七). A stateless-per-turn client suffices.
"""

from __future__ import annotations

from dataclasses import dataclass

from docscout.agent.client import LLMClient
from docscout.agent.parsing import parse_action
from docscout.env.docstore import DocStore
from docscout.env.search_env import EnvConfig, SearchEnv
from docscout.reward.reward import RewardConfig, compute_reward
from docscout.types import QAInstance, Trajectory

SYSTEM_PROMPT = """You are a document-search agent. You must locate the answer to a question in a corpus of natural-language documents using ONLY these tools, one per turn:

- search  : args {query}            -> returns top-k section SNIPPETS (not full text)
- read    : args {idx} or {doc_id, section_id}-> returns one full section. idx = the 1-based number of a candidate in CURRENT CANDIDATES above (easiest).
- expand  : args {doc_id, section_id, direction(left|right)} -> returns a neighbor section
- answer  : args {text, evidence}   -> submit final answer & end (evidence = "doc_id:section_id, ...")

How to work:
- Work STEP BY STEP. For multi-step questions, after you read a numeric value from a section, form a NEW search that includes that value (e.g. the attribute name + the value) to find the next system, then read its relevant section. Repeat until you have the final value.
- NEVER repeat a search query you already issued, and NEVER re-read a section (re-reading returns no new content). If a search/read did not help, CHANGE your query or move to the next step.
- As soon as you have the evidence the question asks for, submit the answer immediately.
- Read as little as possible: searching is cheap (snippets), reading full sections costs tokens.

Reply with EXACTLY one action block:
ACTION: <search|read|expand|answer>
<field>: <value>
...
"""


import os

def _recent_reads(env: SearchEnv, keep: int | None = None) -> str:
    """Build the RECENTLY READ section of the prompt.

    When *keep* is None (the default), reads the optional environment variable
    `DOCSOUT_KEEP_RECENT_READS`.  If that is also unset, defaults to keeping
    **all** entries in `env.read_log` (i.e. no truncation).

    Three-tier memory invariant: `action_log` stays compact (latest ~12
    one-line summaries) regardless of how many reads are kept here.
    """
    if keep is None:
        keep = int(os.environ.get("DOCSOUT_KEEP_RECENT_READS", 1024))

    # When keep is large enough to cover the log, simply slice the whole thing.
    entries = env.read_log[-keep:] if keep else []
    seen, lines = set(), []
    for e in reversed(entries):
        k = (e.doc_id, e.section_id)
        if k in seen:
            continue
        seen.add(k)
        res = env.store.read(e.doc_id, e.section_id)
        if res:
            lines.append(f"[{e.doc_id}.{e.section_id}] {res['content']}")
        if len(lines) >= keep:
            break
    return "\n".join(lines) if lines else "(none yet)"


def build_messages(env: SearchEnv, recent_reads_kept: int | None = None) -> list[dict]:
    cands = "\n".join(
        f"  {i+1}. [{h['doc_id']}.{h['section_id']}] {h['section_title']} — {h['snippet']}"
        for i, h in enumerate(env.current_candidates)
    ) or "  (no search yet)"
    actions = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(env.action_log[-12:])) or "  (none)"
    user = (
        f"QUESTION: {env.question}\n\n"
        f"ACTIONS SO FAR:\n{actions}\n\n"
        f"CURRENT CANDIDATES:\n{cands}\n\n"
        f"RECENTLY READ:\n{_recent_reads(env, recent_reads_kept)}\n\n"
        f"Read budget left: steps {env.cfg.max_steps - env.n_steps}, "
        f"tokens {max(0, env.cfg.max_read_tokens - env.total_read_tokens)}.\n"
        f"Decide the next action."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


@dataclass
class RolloutResult:
    trajectory: Trajectory
    reward: float
    components: dict


def _force_answer_messages(env: SearchEnv, keep: int | None = None) -> list[dict]:
    """Last-step prompt: command the model to answer from what it read."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
             f"QUESTION: {env.question}\n\nYou have run out of steps. "
             f"Using ONLY the evidence you read, you MUST submit your final answer NOW.\n\n"
             f"EVIDENCE READ:\n{_recent_reads(env, keep)}\n\n"
             f"Reply with exactly:\nACTION: answer\ntext: <your answer>"}]


def rollout(
    instance: QAInstance,
    client: LLMClient,
    env_config: EnvConfig | None = None,
    reward_name: str = "ratio",
    reward_config: RewardConfig | None = None,
    recent_reads_kept: int | None = None,
    max_steps_override: int | None = None,
    return_env: bool = False,
):
    """Run one episode. Builds a fresh DocStore per instance (self-contained corpus).

    Args:
        max_steps_override: Static override for all instances (backward-compatible).
            When set, takes precedence over both the config default and the
            dynamic per-instance logic.
    """
    cfg = env_config or EnvConfig()
    if max_steps_override is not None:
        cfg = EnvConfig(max_steps=max_steps_override, max_read_tokens=cfg.max_read_tokens,
                        search_k=cfg.search_k, snippet_token_cost=cfg.snippet_token_cost, rerank=cfg.rerank)

    # Compute per-instance effective max_steps (uses meta.oracle_min_steps when
    # dynamic_max_steps is enabled, unless max_steps_override was explicitly passed).
    effective_max_steps = cfg.effective_max_steps(instance) if max_steps_override is None else max_steps_override
    # Build a fresh config with the effective max_steps resolved. We leave
    # dynamic_max_steps=True so other callers of cfg still get the original
    # behavior, but inject the resolved value directly into the env's config.
    env_cfg = EnvConfig(max_steps=effective_max_steps, max_read_tokens=cfg.max_read_tokens,
                        search_k=cfg.search_k, snippet_token_cost=cfg.snippet_token_cost,
                        rerank=cfg.rerank, dynamic_max_steps=False)  # already resolved
    store = DocStore(instance.docs, rerank=env_cfg.rerank)
    env = SearchEnv(instance, store, env_cfg)
    traj = Trajectory(instance_id=instance.instance_id)

    while not env.done:
        # force-answer at horizon: on the last allowed step, command the model to
        # answer from what it read (so every episode yields an answer -> GRPO signal).
        if env.n_steps >= effective_max_steps - 1:
            messages = _force_answer_messages(env, recent_reads_kept)
        else:
            messages = build_messages(env, recent_reads_kept)
        raw = client.act(env, messages)
        action = parse_action(raw)
        traj.actions.append(action)
        step = env.step(action)
        traj.observations.append(step.observation)

    # finalize trajectory bookkeeping from env
    traj.final_answer = env.final_answer
    traj.final_evidence = [(d, s) for (d, s) in env.final_evidence]
    traj.total_read_tokens = env.total_read_tokens
    traj.n_search, traj.n_read, traj.n_expand, traj.n_steps = env.n_search, env.n_read, env.n_expand, env.n_steps
    traj.terminated_by = env.terminated_by

    reward, components = compute_reward(reward_name, env, instance, reward_config)
    res = RolloutResult(trajectory=traj, reward=reward, components=components)
    return (res, env) if return_env else res