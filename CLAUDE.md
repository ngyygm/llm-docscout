# CLAUDE.md — llm-docscout (DocScout) project guide

## What this is
DocScout: RL-trained NL-document search agent with a **read-budget reward** — the
natural-language analogue of CodeScout (arXiv 2603.17829, cloned at `reference/codescout/`).
Research artifacts: `idea-stage/`, `refine-logs/` (see `自动化实验迭代方案.md` for the
diagnose-before-train methodology that drives the experiment order).

## Compute — this laptop is the HOST, GPU work runs REMOTELY over SSH
- **This laptop** = control host: edit code, download data/models/PDFs, drive experiments.
- **GPU machine** = `ssh root@dev-H200x1` (no password needed). Submit training/inference jobs here only.
  - GPU: **1× NVIDIA H20D (143 GB)**, torch 2.8.0+cu129, CUDA available.
  - **No internet on the GPU box.** Anything that needs downloading (datasets, models, PDFs,
    pip wheels) must be fetched on this laptop first, then `scp`/`rsync` over to the GPU box.
  - Pre-downloaded models live at `/mnt/workspace/zsxdata/local-model/` (Qwen, BAAI, jinaai,
    google, Salesforce/swerank, etc.). Reuse these instead of re-downloading.
  - GPU working directory: `/mnt/workspace/zsxdata/exp-lh/` — push code/configs/data here and
    run jobs from this dir.
- Workflow: develop + stage locally → `scp`/`rsync` to `exp-lh/` → `ssh` run job → pull results back.
- For `/run-experiment` / `/experiment-queue`: target the **remote H20D** (submit via SSH), not local.

## LLM API — host-side data gen / judge / audit (NOT the policy model)
- **GLM-5.2** via OpenAI-compatible endpoint, **up to 400k context** → can feed a whole
  corpus in one call (whole-corpus synth gen, global consistency check, oracle solving).
- Wrapper: `docscout/llm/` — `from docscout.llm import load_config, LLMClient`. Provides
  `complete` / `complete_json` (fence-tolerant) / `batch` (threaded) / `ping`.
- **Key never in git.** Read from `$DOCSCOUT_LLM_TOKEN` or untracked `configs/llm_api.yaml`
  (template: `configs/llm_api.example.yaml`). Smoke: `python -m scripts.llm_smoke --judge`.
- Runs on the **host only** (GPU box has no internet). Uses: v6 synth generation,
  LLM-as-judge answer scoring, MuSiQue closed-book audit. See `DATA_PLAN.md`.

## Conventions
- **Communicate with the user in Chinese** (options + reports).
- **Literature**: use `deepxiv` CLI (layered read: search→brief→head→section); download
  relevant PDFs to `papers/` (real local copies, verify >10KB).
- **Data/env plan**: `DATA_PLAN.md` is the source of truth for dataset cleanup, scale-up,
  v6 heterogeneous synth, real-data expansion, and reward refinement (priority-ordered).
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
