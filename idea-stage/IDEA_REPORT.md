# Idea Discovery Report — Information Acquisition Policy RL

**方向**: 用 RL 训练 ≤2.5B 策略，在 NL 文档库用最小工具集 {search/read/expand/answer} 高效定位+抽取答案，控制"读多少/何时停"。
**日期**: 2026-06-26
**管线**: research-lit(deepxiv) → idea-gen(Claude,GPT-5.4待补) → novelty-check(deepxiv) → review(Claude自评,GPT-5.4待补) → refine+plan
**锚点（用户指定）**: CodeScout (arXiv 2603.17829) — RL 代码定位 agent

## Executive Summary
- 推荐 idea：**DocScout**——把 CodeScout 的 RL 配方（最小工具 + DR.GRPO + 数据/环境工程）迁移到 **NL 文本文档**，并把**读取预算/效率**提为一等奖励目标，刻画 **accuracy-per-read-token 帕累托前沿**。
- 关键证据：CodeScout-1.7B 在 code 上打败 14B（小模型 RL 配方已验证）；NL 文档无 code 的精确结构锚点 → 读取预算在 NL 更关键，而 CodeScout 完全没优化它（max_turns 硬截断）。
- 自评 5/10（borderline-reject，偏增量/受 ALDEN 威胁）；按评审 reframe + 真实 benchmark + ALDEN 切割后可达 6.5–7。
- 推荐下一步：Gate 1 拍板 → Stage 2 克隆 CodeScout 仓库搭最小可跑环境 → R0/R1/R2 pilot。
- ⚠️ 两项待补：①Codex/GPT-5.4 重新认证后补独立对抗评审；②pilot 需配 GPU（needs manual pilot）。

## Literature Landscape（详见 idea-stage/LITERATURE_LANDSCAPE.md）
- **拥挤区**（用户原想"学会何时停搜/读多少/成本感知"已被覆盖）：AutoSearch(2604.17337)、Dynamic-Search-R1(2510.15719)、EVO-RAG(2505.17391)、MBA-RAG(2412.01572)。
- **核心基线族**：Search-R1(2503.09516)、ReSearch(2503.19470)、R1-Searcher(2503.05592)、R-Search(2506.04185)。
- **关键经验**（2505.15117）：format reward 帮助大；中间检索奖励几乎无益；弱引擎→回避检索。
- **本地 PDF 17 篇** 已下载到 `papers/`（含 CodeScout 2603.17829、ReAct、WebGPT、RAG、FLARE 等）。

## Ranked Ideas（详见 idea-stage/IDEA_DRAFT.md + NOVELTY_ASSESSMENT.md）

### 🏆 Idea 1: DocScout — CodeScout 范式的 NL 文档版（读取预算一等）— RECOMMENDED
- Novelty: 中等偏上（位置成立；ALDEN 是头号竞品需切割）
- 自评 Reviewer score: 5/10 → reframe 后 6.5–7
- 差异化三柱：①纯文本 NL × RL 训练（DeepRead/IntrAgent 是 prompting）②读取预算/前沿一等（CodeScout accuracy-only）③≤2.5B 小模型配方
- 下一步：Stage 2 实现 → /auto-review-loop

### Idea 2: FRONTIER — accuracy-per-read-token 帕累托前沿评测（BACKUP/已并入 DocScout 贡献③）

### Idea 3: RatioRL — efficiency-ratio 抗作弊奖励（BACKUP/已并入 DocScout 贡献②）

## Eliminated Ideas
- I6 CalibStop：与 AutoSearch"中间答案判充分"重叠。
- I7 AttrReward：与 R-Search evidence 奖励重叠。
- I8 MemCompact：纯工程，非 novelty。
- 原 ReadGrind(I1 抽象合成)：被 DocScout（CodeScout 锚定版）取代，更扎实。

## Refined Proposal（详见 refine-logs/FINAL_PROPOSAL.md）
- Problem Anchor（冻结）、Method Thesis（已 reframe 为 finding+method）、4 件套贡献、reward 三变体、DR.GRPO、合成+真实数据。

## Experiment Plan（详见 refine-logs/EXPERIMENT_PLAN.md）
- 4 claims ↔ 4 实验；必跑消融 reward/action/scale/doclen/hack；6 基线（含 Search-R1/AutoSearch 文档适配 + ALDEN 切割）；主指标帕累托前沿 + 明确协议；首批 R0/R1/R2 run。

## Next Steps
- [ ] **Gate 1（现在）**：用户拍板 DocScout（或选 backup / 调整）。
- [ ] Stage 2：克隆 github.com/OpenHands/codescout + HF collection 作参考 → 搭最小可跑环境（合成语料 + ≤2.5B + DR.GRPO + 三 reward 变体）。
- [ ] Stage 3：R0/R1/R2 pilot（需配 GPU）。
- [ ] Stage 4：/auto-review-loop（含 Codex 重认证后的 GPT-5.4 评审）。
- [ ] Codex/GPT-5.4 重新认证 → 补 Phase 3 交叉验证 + Phase 4 独立对抗评审。
