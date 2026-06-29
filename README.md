# DocScout — RL-trained NL-document search agent with a read-budget reward

> The natural-language-document analogue of [CodeScout](https://github.com/OpenHands/codescout) (arXiv 2603.17829).
> CodeScout trains a small RL agent to localize code with a Unix terminal. DocScout trains a small (≤2.5B) RL agent to locate & extract answers in **natural-language documents** — and, unlike CodeScout, treats **reading budget / efficiency as a first-class reward objective**.

Research direction, idea validation, novelty, review, and experiment plan live in
[`idea-stage/`](idea-stage/) and [`refine-logs/`](refine-logs/). See
[`refine-logs/FINAL_PROPOSAL.md`](refine-logs/FINAL_PROPOSAL.md) for the method and
[`refine-logs/DATA_DESIGN.md`](refine-logs/DATA_DESIGN.md) for the data plan.

## Status (Stage 2)
- ✅ Self-contained, **CPU-runnable** core: synthetic data → DocStore → SearchEnv → reward engine → rollout → metrics.
- ✅ R0 smoke test passes; 7/7 unit tests pass; results save to JSON.
- ⏳ GRPO training adapter written (GPU, framework-light); needs GPU + `pip install -e .[train]`.
- ⏳ Grounded reverse-construction data pipeline (MuSiQue/HotpotQA/DocScope + necessity/sufficiency validation) — next, per DATA_DESIGN.md.

## The core idea
Agent tools (discrete type, generated args): `search(query)→snippets`, `read(doc,section)→full section`, `expand(doc,section,direction)→neighbor`, `answer(text,evidence)→end`. Reward:

```
R_add    = answer + w·evidence − λ·read_tokens                  (additive cost)
R_redund = answer + w·evidence − β·redundancy − λ·read_tokens   (ALDEN-flavored)
R_ratio  = answer + w·evidence − γ·(1 − efficiency_ratio)       (ours, anti "read everything")
          efficiency_ratio = gold_tokens_in_context / total_read_tokens
```
Primary metric: **accuracy-per-read-token Pareto frontier**. `evidence` is computed over
*committed reads* (READ/EXPAND) — glimpsing a snippet is cheap exploration, not verified evidence.

## Layout
```
docscout/
  types.py                 # Section/Document/QAInstance/Action/Trajectory
  data/synth_generator.py  # synthetic NL doc→section + QA + section-locatable gold evidence
  env/docstore.py          # BM25 snippet search + read + expand
  env/search_env.py        # 4-action env, three-tier memory state, committed-vs-snippet accounting
  reward/answer_scoring.py # normalized EM (+ partial)
  reward/reward.py         # @reward registry: additive / redundancy / ratio
  agent/parsing.py         # lenient ACTION-block parser
  agent/client.py          # StubClient (CPU heuristic) + OpenAIClient (vLLM)
  agent/rollout.py         # compact-memory prompt + episode loop
  train/grpo.py            # framework-light GRPO reference (GPU; lazy torch)
configs/                   # env + 3 reward variants (yaml)
scripts/run_rollout.py     # R0 smoke (CPU)
scripts/run_train.py       # GRPO launcher (GPU)
tests/test_core.py         # 7 unit tests
papers/                    # 17 downloaded reference PDFs (deepxiv/arxiv)
reference/codescout/       # CodeScout implementation (cloned for reference)
```

## Run

```bash
# 1) R0 smoke on CPU (no GPU, no model) — verifies full plumbing + reward signal
python -m scripts.run_rollout --n-instances 30 --verbose --out results/r0_smoke.json

# 2) unit tests
python tests/test_core.py          # or: pip install pytest && pytest tests/

# 3) generate a synthetic split
python -m docscout.data.synth_generator --n-instances 5000 --out data/synth/train.json

# 4) GRPO training on GPU (needs pip install -e .[train] + a served/loaded Qwen3-1.7B)
python -m scripts.run_train --env configs/env_default.yaml \
    --reward configs/reward_ratio.yaml --max-steps 500
```

## CodeScout reference mapping
| DocScout | CodeScout (reference/codescout/) |
|---|---|
| `docscout/reward/reward.py` (`@reward` registry) | `src/rewards/multiturn.py`, `src/rewards/file_localization/` |
| `docscout/env/search_env.py` (read budget) | `src/metrics/efficiency_metrics.py` (note: theirs = LLM tokens; ours = doc-read tokens) |
| `docscout/data/synth_generator.py` | `src/generator/code_search_generator.py`, `src/build_dataset.py` |
| `docscout/train/grpo.py` | `src/train.py`, `src/async_trainer.py` (SkyRL, DR.GRPO) |
| `configs/reward_*.yaml` | `configs/reward_config_1.7b.yaml` |

For the CodeScout-faithful SkyRL path, adapt `reference/codescout/src/train.py` and register
`docscout.reward.compute_reward` as the reward hook (it consumes the rollout env's read_log).

## Known limitations / next
- R0 uses the heuristic `StubClient`; real rollouts need a served ≤2.5B model.
- GRPO reference recomputes episode logprobs approximately (per README note in `train/grpo.py`); the SkyRL path is preferred for the final recipe.
- Data is synthetic; the grounded MuSiQue/HotpotQA/DocScope reverse-construction pipeline (`refine-logs/DATA_DESIGN.md`) is the next build.
- Frontier measurement protocol (λ-sweep / budget-sweep) to be added before real results.
