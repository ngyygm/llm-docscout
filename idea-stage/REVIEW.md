# 外部对抗评审 — Phase 4（⚠️ Claude 自评；GPT-5.4 独立评审待 Codex 重新认证）

> 角色：NeurIPS/ICML 资深 AC，刻意刁难。评审对象：DocScout（CodeScout 式 RL agent → NL 文档 QA，读取预算/效率一等奖励，≤2.5B）。

## 评分：**5/10**（borderline-reject）——"有意思但偏增量、受 ALDEN 威胁；需更锋利的 claim + 真实 benchmark 才能上岸"。经下方 reframe + 补强后可达 **6.5–7/10**（borderline-accept）。

## 最大威胁判定
**ALDEN (2510.25668) 是头号威胁**——它已是"RL + 长文档导航 + token/turn 级 reward 防冗余"。当前差异化（"VLM vs 纯文本"、"防冗余 vs 读取预算前沿"）**可能被审稿人判为增量**。仅靠"换到文本域"不够，必须证明 **read-budget/efficiency-ratio reward 得到与 ALDEN 冗余-reward 质不同的前沿结论**。

## Top 5 弱点（按严重度）+ 最小修复

**W1【严重·新颖性】ALDEN 近似度太高。**
→ 最小修复：把 ALDEN 式"冗余惩罚 reward"和"加性 −λ·token"都作为 reward 变体纳入消融；**核心贡献定位为"reward 设计 + 前沿刻画"而非"ALDEN-on-text"**。证明 efficiency-ratio 在同等准确率下严格改善前沿。

**W2【严重·贡献稀释】"把 CodeScout 搬到 NL"像应用文而非方法文。**
→ 最小修复：把 thesis 从"迁移"改为**实质性经验 claim**：*在长 NL 文档上，文档内读取是主导成本，search-depth-RL(Search-R1/AutoSearch) 与固定预算 RAG 处于被支配前沿；read-budget 策略支配它们。* CodeScout 只是"配方使能者"，不是贡献。

**W3【高·指标正当性】"accuracy-per-read-token 帕累托前沿"是否站得住？** 审稿人会问：为什么是 read-token 而非 wall-clock/工具调用数/$/上下文长度？
→ 最小修复：论证 read-token = 长文档场景的**上下文污染成本**（主导项）；同时**附带报告 #tool-calls 与 latency 作次轴**；给出明确的前沿测量协议（扫 reward 的 λ 或扫预算上限，画前沿）。

**W4【高·benchmark 合法性】无数据集；纯合成语料=自评玩具数据。**
→ 最小修复：**至少落地一个真实 benchmark**（带 section 可定位证据）：候选 HotpotQA/MuSiQue（段落作 section）、NaturalQuestions+Wikipedia sections、或技术手册 QA；**合成语料仅用于隔离消融**。审稿人一定会要求真实数据。

**W5【中·奖励作弊】read-budget 易诱发"不读就幻觉"或"读一片就猜"。**
→ 最小修复：失败模式分析；证明紧预算下准确率不崩；reward 变体对比。

## 实验设计批注
- "accuracy-per-read-token 帕累托前沿"作为主指标**可成立但需精确定义测量协议**；单点 accuracy 排序会掩盖成本差异（这点是好卖点）。
- 必跑消融：(a) reward 变体 {加性 / ALDEN冗余 / efficiency-ratio}；(b) 动作空间 {去 expand / 去 read 只 search}；(c) 模型规模 {1.5B/2.5B/7B}；(d) 文档长度档位。
- 审稿人必要求的缺失基线：**把 Search-R1/AutoSearch 适配到文档设置**并显示其被支配（否则"为何不是 AutoSearch 特例"答不硬）。
- 必须有 ALDEN 的文本适配对照（或至少其 reward 变体）。

## 最能提分的一处改动
> **把论文从"CodeScout→NL 迁移"重写为"长 NL 文档上，文档内读取是主导成本；我们证明 search-depth-RL 与固定预算 RAG 在 accuracy-per-read-token 前沿上被 read-budget-aware 策略支配，并用 efficiency-ratio reward 刻画其机制"。**
> 即：**finding + method** 论文，CodeScout 是配方来源不是卖点。

## 结论性意见（给 Gate 1）
- 方向**可行且值得做**，但当前定位偏增量；按 W1–W4 重构后是合格的 top-venue 投稿。
- 必须解决：①ALDEN 切割 ②真实 benchmark ③前沿测量协议 ④Search-R1/AutoSearch 文档适配基线。
- 建议 Stage 2 实现时**先搭最小可跑环境（合成语料 + ≤2.5B + GRPO + 三 reward 变体）**，跑通 pilot 再上真实 benchmark。
