# Paper Plan — DocScout (AAAI)

## Title
**DocScout: Read-Budgeted Reinforcement Learning for Natural-Language Document Search Agents**

## One-line thesis
We port CodeScout's minimal-tool RL recipe to natural-language documents and make
**reading budget / efficiency a first-class reward objective**, optimizing the
**accuracy-per-read-token frontier**; we show (i) the environment is solvable but
difficulty-sensitive, (ii) plain SFT suffers a *stop-failure* pathology that
blocks RL, and (iii) an answer-balanced SFT restores a strong GRPO learning signal.

## Claims ↔ Evidence matrix
| Claim | Evidence (file) |
|---|---|
| C1 Retrieval is sufficient; agent's job = selection+stop | retriever Recall@5=0.847 MuSiQue (`results/retriever_recall_musique.json`) |
| C2 Task is solvable but difficulty-sensitive | oracle synth-easy 0.85 vs MuSiQue multi-hop 0.225 (`results/_oracle_synth.json`, `results/oracle_evidence_musique.json`) |
| C3 Middle-difficulty substrate gives oracle−RAG headroom | synth-v2 oracle 0.748, RAG 0.609 (`results/v2_oracle_vllm.json`, `v2_rag_vllm.json`) |
| C4 Untrained agent is far below RAG; failures are stop/selection | prompted 0.110 + failure breakdown (`results/v2_prompted.json`) |
| C5 SFT stop-failure blocks RL; answer-balancing fixes it | SFT-v1 0% answer vs SFT-v2 48.5% answer, group-advantage 0.316 (`results/rl_signal.json`, `rl_signal_sft2.json`) |
| C6 RL improves accuracy-per-read-token frontier | [TBD-RL] (GRPO convergence pending) |

## Sections (8)
1. Abstract
2. Introduction (motivation: RAG context bloat; CodeScout anchor; read-budget thesis; contributions)
3. Related Work (ReAct, WebGPT, RAG, Search-R1/ReSearch, AutoSearch/Dynamic-Search-R1 [search-depth], DeepRead/IntrAgent [prompting], ALDEN [VLM RL] — positioning)
4. Method (env + tools; reward variants add/redund/ratio + efficiency-ratio; SFT demos; GRPO)
5. Experimental Setup (synth-v2 difficulty stages; MuSiQue/HotpotQA grounded; metrics: accuracy, evidence, read-tokens, Pareto frontier; Qwen3-1.7B; 3×3090)
6. Results (Phase-1 solvability; Phase-2 baselines + failure analysis; Phase-3 SFT stop-failure + RL signal; [RL frontier TBD])
7. Analysis & Discussion (the stop-failure pathology; when RL helps; threats to validity)
8. Conclusion + Limitations + [TBD-RL] Future Work

## Figures/Tables
- **T1**: Phase-1 retriever recall + oracle solvability (synth vs MuSiQue). [data ready]
- **T2**: Phase-2 baselines + failure breakdown on synth-v2. [data ready]
- **F1**: bar chart — oracle/RAG/prompted accuracy on synth-v2 (with direct vs paraphrased split). [data ready]
- **F2**: SFT stop-failure → fix: answer-rate & group-advantage-signal, v1 vs v2. [data ready]
- **F3 (TBD-RL)**: accuracy-per-read-token Pareto frontier across reward variants. [pending GRPO]

## Notes
- Compile class: use portable `article` (AAAI `aaai25.sty` not offline) — swap for submission.
- RL convergence numbers are [TBD-RL] placeholders; all other numbers are real.
- GPT-5.4 adversarial review pending Codex re-auth.
