# CLAUDE.md — llm-docscout (DocScout) project guide

## What this is
DocScout: RL-trained NL-document search agent with a **read-budget reward** — the
natural-language analogue of CodeScout (arXiv 2603.17829, cloned at `reference/codescout/`).
Research artifacts: `idea-stage/`, `refine-logs/` (see `自动化实验迭代方案.md` for the
diagnose-before-train methodology that drives the experiment order).

## Compute — run experiments LOCALLY
- **GPU: local**, 3 GPUs: `0` = RTX 3090 Ti (24GB), `1`,`2` = RTX 3090 (24GB each).
- CUDA available via torch; 125GB RAM.
- For `/run-experiment` / `/experiment-queue`: use **`gpu: local`**. No SSH/rsync needed.
- Policy model ≤2.5B (Qwen3-1.7B / 2.5B) fits comfortably per GPU.

## Conventions
- **Communicate with the user in Chinese** (options + reports).
- **Literature**: use `deepxiv` CLI (layered read: search→brief→head→section); download
  relevant PDFs to `papers/` (real local copies, verify >10KB).
- Follow the experiment order in `refine-logs/自动化实验迭代方案.md`: Phase-1 environment
  solvability (oracle evidence, retriever Recall@k, oracle path) BEFORE any RL training.
- One-factor-at-a-time; failure-classify every run (retrieval/selection/reading/stopping/answer/hack).
- Reward: `docscout/reward/reward.py` variants `additive`/`redundancy`/`ratio`; iterate V0→V4.

## Target
AAAI submission. Use `/research-pipeline` end-to-end; `/paper-writing` with `venue: AAAI`.
Note: Codex/GPT-5.4 (auto-review-loop, paper improvement) needs `codex login` re-auth.

## Quick run
```
python -m scripts.run_rollout --n-instances 30 --verbose --out results/r0_smoke.json   # CPU smoke
python tests/test_core.py                                                              # 7 unit tests
python -m scripts.run_train --env configs/env_default.yaml --reward configs/reward_ratio.yaml --max-steps 500  # GPU GRPO
```
