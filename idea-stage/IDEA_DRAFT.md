# Idea Draft — Phase 2（Claude 生成；GPT-5.4 对抗评审待 Codex 重新认证后补做）

> 输入：RESEARCH_BRIEF.md + idea-stage/LITERATURE_LANDSCAPE.md。
> 硬约束已强制：每个 idea 必须回答"为什么不是 AutoSearch(2604.17337)/Dynamic-Search-R1(2510.15719) 的特例"。

## 关键再框定（决定一切新颖性判断）

现存 RL-agentic-search 工作（Search-R1/ReSearch/AutoSearch/Dynamic-Search-R1）几乎全部跑在 **web 搜索**上：每次 `search(query)` 返回 top-k **短段落**，"成本"= **检索次数/检索 token**，agent 控制的是**搜索广度/深度**。

真正相对空白的是 **单篇长文档内部的读取粒度控制**：agent 决定的不只是"要不要再搜一次"，而是"这篇文档我到底需要读多细"。此时真正的成本是 **被读进上下文的 token（context 污染）**，而非检索次数。这个区别只在**长结构化文档**（手册/政策/技术规格）场景才显著。

→ 任何 idea 的护城河都应锚定在 **"文档内读取粒度 + 读预算"**，而不是 web-search-depth。

---

## 候选 idea（9 个）

### I1. ReadGrind — 长文档内读取粒度的 RL 策略 🏆
- **Thesis**：在长结构化文档上，控制**文档内读取粒度**（snippet→read(section)→expand(neighbor)）的 RL 策略，在"准确率-per-读取-token"前沿上优于固定预算 RAG 与 search-depth-only RL（AutoSearch）。
- **Core mechanism**：env = doc→section 树；动作 `search/read(section)/expand(dir)/answer`；reward = EM − λ·tokens_read；用 efficiency-ratio 项抗"全读再答"。读取量=真正进入上下文的 token。
- **Closest + diff**：AutoSearch(2604.17337) 控**搜索深度**；本 idea 控**单文档内读取粒度**。Dynamic-Search-R1(2510.15719) 成本是检索 token；本 idea 成本是**读取入上下文 token**，动作空间含 expand。
- **为何不是 AutoSearch 特例**：动作空间不同（read/expand vs search）、成本量纲不同（tokens-read vs #search）、setting 不同（长文档导航 vs web 段落 QA）。AutoSearch 在 web 段落上"读粒度"几乎无杠杆，本 idea 的杠杆正在于此。
- **最小实验**：合成多 section 长文档语料（policy/技术手册风格，植入可定位 QA）；基线 fixed top-k RAG / FLARE / Search-R1-style / AutoSearch-style；主指标 accuracy@token-read 帕累托前沿。
- **可行性(≤2.5B+合成语料)**：4/5（env 与合成数据可控；≤2.5B 可 RL）。
- **新颖性**：4/5（overlap 风险：需 Phase 3 排查"长文档导航+RL+读预算"是否已被做）。
- **Kill 条件**：已存在论文做"长文档内 read/expand + 读预算 reward"的 RL。

### I2. FRONTIER — accuracy-per-token-read 帕累托前沿作为主评测
- **Thesis**：现有 agentic-RAG 只报单点准确率；改用"准确率-per-读取-token"帕累托前沿作主指标，揭示方法排序在单点 vs 前沿下反转。
- **Closest + diff**：Flare-Aug(2502.12145) 有 accuracy-cost 旋钮；MBA-RAG(2412.01572) 有 accuracy+cost 奖励。本 idea 把**前沿刻画**作为主贡献 + 受控长文档导航 benchmark。
- **为何不是特例**：是评测/benchmark 贡献，与 AutoSearch 正交；可作为 I1 的评测骨架。
- **可行性**：4/5；**新颖性**：3/5（"只是 benchmark"单发难冲顶会，强作为 I1 配角）。
- **Kill**：已有标准化的"读取 token 前沿"benchmark。

### I3. RatioRL — 效率比奖励抗 read-everything 作弊
- **Thesis**：加性成本惩罚(−λ·tokens) 诱发"全读再答"或"少读但答错"的退化；用 **efficiency-ratio 成形优势**（gold-evidence tokens / total tokens read，经归因计算）在同等准确率下得到严格更优前沿。
- **Closest + diff**：Search-P1(2602.22576) path-centric shaping；EVO-RAG(2505.17391) 七因子含 efficiency。本 idea 聚焦**比值项 + 基于证据归因**的抗作弊机制。
- **为何不是特例**：奖励设计贡献，可作为 I1 的方法核心。
- **可行性**：3/5（证据归因需 gold section，合成语料可控）；**新颖性**：3/5（奖励 shaping 拥挤，需比值特异性）。
- **Kill**：比值项相对加性惩罚在前沿上无显著差异。

### I4. ReadOrSkip — 读前必要性元认知门控
- **Thesis**：把"自适应检索少用反而好"(2602.07213) 变成显式门：读 section 前用小预测头判断"本次读取对答对是否必要"（answerability/gain），RL 联合优化门+读策略。
- **Closest + diff**：GRIP(2604.11407) 用 control token 自触发检索；本 idea 门控的是**读取**而非检索，且显式建模必要性。
- **为何不是特例**：门控对象是 read 粒度，区别于 search 触发。
- **可行性**：3/5；**新颖性**：3/5（经验研究说中间奖励几乎无益，需谨慎）。
- **Kill**：门控收益被 format reward 吞没。

### I5. SmallScale-Frontier — ≤2.5B 的读取效率前沿更陡
- **Thesis**：≤2.5B 上 accuracy-per-token-read 前沿比 7B 更陡，读取粒度控制在更小模型上收益更大（"读得更少更精"对小模型更关键）。
- **Closest + diff**：经验研究(2505.15117) 说规模收益递减；本 idea 聚焦**效率前沿随规模**的变化。
- **为何不是特例**：scaling-analysis 贡献，正交于 AutoSearch。
- **可行性**：3/5（需多尺度，算力约束）；**新颖性**：3/5。
- **Kill**：小模型前沿并未更陡。

### I6. CalibStop — 证据充分性校准停止
- **Thesis**：训练 agent 发出校准的"证据已充分"信号作 stop/answer 触发，奖励同时罚早停与晚停。
- **Closest + diff**：AutoSearch 用中间答案判充分性；GRIP 用 [SOLVED] token。重叠中等。
- **为何不是特例**：**较弱**——AutoSearch 已做"中间答案判充分"。🟡 倾向淘汰。
- **可行性**：3/5；**新颖性**：2/5。
- **Kill**：与 AutoSearch 充分性机制无本质区别。

### I7. AttrReward — 基于证据归因的稀疏奖励
- **Thesis**：用 section 级证据归因把稀疏 outcome reward 变成稠密"读到正确 section 才给分"，但配合 anti-hack 比值避免全读刷分。
- **为何不是特例**：R-Search(2506.04185) 已有 evidence-quality 奖励；**重叠高**。🔴 倾向淘汰。
- **可行性**：4/5；**新颖性**：2/5。

### I8. MemCompact — 三层记忆 + snippet-only 观测的工程研究
- **为何不是特例**：纯工程实现，非 novelty。🔴 淘汰（并入 I1 实现细节）。

### I9. NoisyIndex — 弱/噪声检索引擎下的鲁棒读策略
- **Thesis**：经验研究(2505.15117) 指出弱引擎→检索回避；研究在**噪声索引**下读粒度策略如何补偿检索噪声。
- **为何不是特例**：鲁棒性角度，正交。但较 niche。
- **可行性**：3/5；**新颖性**：3/5。

---

## 严苛筛选 + 排序

按 (novelty × feasibility × signal) 排：

1. 🏆 **I1 ReadGrind**（4×4×高）— 最干净的单点贡献：setting+动作空间+读预算，与 web-search-depth 族正交。
2. 🏆 **I3 RatioRL**（3×3×中）— 作为 I1 的奖励核心，单发偏弱，合成后强。
3. 🏆 **I2 FRONTIER**（3×4×中）— 作为 I1 的评测骨架，单发偏 benchmark。

**淘汰**：I6（与 AutoSearch 充分性机制重叠）、I7（与 R-Search evidence 奖励重叠）、I8（纯工程）。

## 🎯 RECOMMENDED SYNTHESIS（推荐合成）

把 I1 + I2 + I3（+ I5 作 scaling 分析配角、I4 作可选消融）合成**一个连贯贡献**：

> **"Read-Budgeted Document Navigation: 把信息获取建模为'文档内读取粒度控制'，在 token-read 预算下的 RL 策略。"**
> 贡献四件套：
> (a) **新 setting/env**：长结构化文档导航，section 可定位证据，成本=读入上下文 token（区别于 web search-depth/检索 token）；
> (b) **动作空间**：search / read(section) / expand(neighbor) / answer；
> (c) **效率比奖励**：抗"全读再答"作弊，对照加性惩罚做消融；
> (d) **评测**：accuracy-per-token-read 帕累托前沿作主指标，对照 fixed-budget RAG / FLARE / Search-R1-style / AutoSearch-style；
> (e)（配角）≤2.5B 小模型前沿分析。

**一句话差异化**："AutoSearch 学何时停止**搜索**（web 段落，成本=#检索）；本工作学何时停止**读取**（长文档，成本=读入 token）。"

**Phase 3 必查 kill 条件**：是否已有 2025-2026 论文做"长文档内 read/expand + 读预算 reward 的 RL"。下一阶段用 deepxiv 专门猎杀这个威胁。
