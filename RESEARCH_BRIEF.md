# Research Brief — Information Acquisition Policy RL

> Distilled from the user's system-design discussion. This brief is the primary
> context for the idea-discovery pipeline (literature → ideas → novelty → review).

## Direction (one line)

Train a **small (≤2.5B)** language-model policy, via **RL**, to efficiently locate
and extract the correct answer from a corpus of natural-language documents using a
minimal tool set (`search` / `read` / `expand` / `answer`). The policy must learn
**how much to read** and **when to stop**.

## Problem statement

Standard RAG retrieves a fixed top-k context once and hands it to the generator.
This (a) bloats the context with irrelevant passages, and (b) gives the model no
control over retrieval depth. We want an agent that, like a human scanning a
manual, decides step-by-step: what to look up, which snippet to open, whether to
expand to neighbor sections, and when it has gathered enough evidence to stop —
**minimizing total tokens read while preserving answer correctness.**

## Working hypothesis (to be sharpened by literature search)

A small RL-trained acquisition policy can match or exceed fixed-budget RAG on
accuracy while reading substantially fewer tokens, because it learns a
**stop / read-depth policy conditioned on partial evidence** rather than a fixed
retrieval budget. The precise novel contribution is **deliberately left open** —
see *Constraints → Novelty*.

## Proposed system (MVP, from the design discussion)

- **Tools** (discrete *types*, generated *args*):
  - `search(query)` → top-k **snippets only** (not full text)
  - `read(doc_id, section_id)` → one full section + neighbor availability
  - `expand(doc_id, section_id, direction)` → adjacent section (`left`/`right`) — dynamic chunking via neighbor expansion, **not** re-chunking
  - `answer(text, evidence)` → terminate episode
- **Document store:** structured into stable `doc → section` units so every read is locatable and reward-assignable.
- **Observation discipline:** the agent never sees the whole corpus. Per step it sees: question + compact action log + current candidate snippets + a small evidence-notes buffer. Three-tier memory:
  1. **Question** (permanent)
  2. **Action log** (compact: `search(q)→8 hits`, `read(doc_12.sec_3)`, …)
  3. **Current best evidence** (small — only the most relevant snippet/section text kept)
- **Reward:** `reward = correctness − search_cost`
  - `+10` final answer correct; `+3` cited evidence truly supports the answer
  - `−0.05` per `search`, `−0.10` per `read`, `−0.20` per 1k tokens read
  - `−5` wrong answer, `−3` wrong evidence, `−2` timeout (> max steps)
  - **Anti-hacking efficiency term:** `efficiency = gold-evidence tokens / total tokens read`, to stop the policy from "read everything then answer" (which a naive `+5 per correct section` reward invites).

## Constraints

- **Policy model size:** **≤2.5B** (1.5B or 2.5B; **not 7B**). Must be RL-trainable on available GPUs.
- **Compute:** GPU servers available (exact config not yet wired into this repo).
- **Data:** **No dataset yet.** Need to auto-generate / synthesize a document corpus + `(question, gold_answer, gold_evidence_section)` triples, **or** adapt an existing open QA-over-docs benchmark that exposes attributable section-level evidence.
- **Novelty:** **Open.** Run a broad literature-driven search; let novelty-check pick the most defensible contribution rather than anchoring early. Candidate angles to *evaluate* (not pre-select):
  1. **Learn-to-stop efficiency** — near-optimal stop/read-depth policy beats fixed-budget RAG on accuracy-per-token.
  2. **Anti-reward-hacking retrieval RL** — efficiency-shaped rewards prevent "read everything" exploits.
  3. **Expandable-chunk RL policy** — `read`/`expand` as RL actions beats fixed chunks for multi-section evidence.
- **Pilots:** **Not feasible in this repo yet** (no dataset, no GPU server config). Validate ideas on paper + tiny CPU-scale checks; mark every pilot as "needs manual pilot."

## Known related work (anchors, not exhaustive — to expand via literature search)

- **ReAct** (Yao et al., 2210.03629) — reasoning → action → observation.
- **WebGPT** (Nakano et al., 2112.09332) — browser-assisted QA with human feedback.
- **RAG** (Lewis et al., 2005.11401) — retrieval-augmented generation.
- *To map:* RL for retrieval/tool-use (e.g. RLHF/GRPO over tools, Search-R1, ReSearch, agentic retrieval, adaptive reading, FLARE-style forward-looking active retrieval, token-budgeted retrieval).

## Non-goals

- Not a general web-search agent.
- Not scaling to 7B+.
- Not pure QA-accuracy optimization — efficiency is a **first-class** objective.

## Open questions the pipeline must resolve

1. What is the cleanest, most defensible novel contribution given existing work?
2. Which dataset/benchmark (or synthetic design) gives controllable, **section-locatable** evidence for reward computation?
3. Which RL algorithm (GRPO / PPO / REINFORCE) best fits a small tool-using policy with a `correctness − cost` reward?
4. How do we benchmark "accuracy-per-token-read" fairly against fixed-budget RAG baselines?

## Status

- **Date:** 2026-06-26
- **Repo state:** empty (greenfield).
- **Current stage:** Stage 1 — Idea Discovery (research-pipeline Gate 1 will pause for a human pick).
