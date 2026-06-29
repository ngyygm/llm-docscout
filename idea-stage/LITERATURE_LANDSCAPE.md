# 文献景观图 — Information Acquisition Policy RL

> Phase 1 输出。来源：6 组 deepxiv 并行检索（57 篇去重）+ 16 篇真实下载到 `papers/`。
> 检索主题：RL-for-retrieval / agentic search / adaptive reading / token-budget / active retrieval。

## 0. 一句话结论（对新颖性至关重要）

**用户最初设想的"用 RL 学会何时停搜 / 读多少 / 成本感知"这个卖点，已被 2025–2026 年的一批工作大量覆盖。**
现存的强基线包括 AutoSearch（最小充分搜索深度）、Dynamic Search-R1（成本感知优势函数）、EVO-RAG（含 efficiency 的七因子奖励）、MBA-RAG（精度+成本联合奖励）。
→ **新颖性必须收敛到更锐的角度**（见 §4 gap），否则会撞 AutoSearch / Dynamic-Search-R1。这个判断交给 Phase 2 idea-gen + Phase 3 novelty-check 最终定夺。

## 1. 子方向与代表论文（★ = 已下载 PDF 到 papers/）

### A. RL 训练搜索/推理交错 agent（核心基线族）
- ★ **Search-R1 (2503.09516)** — RL 让 LLM 边推理边发搜索 query；retrieved-token masking；outcome reward；比 RAG +20–24%。7 个 QA 数据集标杆基线。
- ★ **ReSearch (2503.19470)** — GRPO 训练多跳推理+搜索，无需推理步监督；+8.9–22.4%。
- ★ **R1-Searcher (2503.05592)** — 两阶段：retrieve-only reward → answer reward；HotpotQA 上超 GPT-4o-mini 达 48%。
- ★ **R-Search (2506.04185)** — 多奖励（answer / evidence / format），把检索文档蒸馏成结构化证据塞进推理链。
- ★ **COSEARCH (2604.17555)** — 联合训练推理与文档排序。

### B. 效率 / 自适应深度 / 成本感知（与用户设想直接重叠 —— 拥挤区）
- ★ **AutoSearch (2604.17337)** — ⚠️**最直接竞品**：RL 动态决定"最小充分搜索深度"，奖励达到最小深度、惩罚 over-search；建立 accuracy-efficiency 权衡。
- ★ **Dynamic Search-R1 / Cost-Aware (2510.15719)** — ⚠️成本感知优势函数：memory-bound（最小化总 token）与 latency-bound；16–20% 延迟下降 +5% EM。
- ★ **EVO-RAG / Curriculum RL (2505.17391)** — 七因子步级奖励（relevance/redundancy/efficiency/correctness），检索深度降 15%。
- **MBA-RAG (2412.01572)** — 多臂老虎机，联合精度+成本奖励，惩罚高成本策略，检索开销降 20%。
- **Flare-Aug (2502.12145)** — 用户可控 accuracy-cost 权衡（参数 α）。
- ★ **s3 (2505.14146)** — Gain-Beyond-RAG (GBR) 奖励；2.4k 样本即可，解耦 search 与 generation。
- ★ **Search-P1 (2602.22576)** — path-centric reward shaping（self-consistency + reference-alignment），解决稀疏奖励；+7.7%。

### C. 何时触发 / 何时停（control-token / active-retrieval 族）
- ★ **GRIP / Retrieval-as-Generation (2604.11407)** — token 级控制符 `[RETRIEVE]/[INTERMEDIARY]/[ANSWER]/[SOLVED]`，自触发检索与终止。
- ★ **FLARE (2305.06983)** — 主动检索鼻祖：生成临时下一句→低置信则检索。
- ★ **Unified Active Retrieval (2406.12534)**、**RARE (2412.02830)**、**Decide-Then-Retrieve (2601.03908)** — decide-when-to-retrieve 系列。

### D. token 预算 / 读取量（相邻，但多为推理长度而非检索读取）
- ★ **TALE / Token-Budget-Aware (2412.18547)** — 动态调整 CoT token 预算；token 用量降 67%。
- **Budget AI Researcher (2506.12317)** — RAG chain 的预算约束。

### E. 经验研究与综述（设计要点来源）
- ★ **Empirical Study RL reasoning-search (2505.15117)** — ⚠️关键经验：①**format reward 显著提升最终性能；中间检索奖励几乎无益**；②通用 LLM > 推理专用 LLM（早期指令遵循更好）；③模型规模收益递减；④**搜索引擎质量关键塑造 RL 动态**（弱引擎→回避/低效检索）。
- ★ **Survey: RL-based Agentic Search (2510.16724)** — 首个 RL-agentic-search 系统框架综述（角色/优化/评测/应用）。
- **Adaptive Retrieval helps — but mostly if not used (2602.07213)** — ⚠️警示：自适应检索"少用反而好"（GSM8K 7%, MATH 38.8%），自我评估是关键元认知。

### F. 锚点谱系（用户引用）
- ★ ReAct (2210.03629)、★ WebGPT (2112.09332)、★ RAG (2005.11401)。

## 2. 现存工作的共同假设（= 破局点所在）

1. **检索单元 = "web 搜索 query → 返回 top-k 整段"**，黑盒化。几乎所有 B/C 族都优化"搜索次数/深度"或"检索 token 总量"。
2. **cost 多为加性惩罚**（−0.1/search, −tokens），少有显式**反作弊效率比**。
3. **模型多为 7B+**（Search-R1 用 Qwen2.5-7B）；≤2.5B 的文档导航策略效率前沿少有系统研究。
4. **评测多在 web-search QA**（HotpotQA/NQ/TriviaQA 等），少有"文档内部 section 级导航 + 可定位证据"的受控环境。

## 3. 与用户 RESEARCH_BRIEF 的对照

| 用户设想 | 现状 | 风险 |
|---|---|---|
| 学会"何时停搜" | AutoSearch / Dynamic-Search-R1 已做 | 🔴 高度重叠 |
| reward = correctness − cost | EVO-RAG / MBA-RAG / Cost-Aware 已做 | 🔴 重叠 |
| read/expand 邻居段落（文档内动态切块） | 极少针对**文档内 section 级读取粒度**做 RL | 🟢 相对空白 |
| 反奖励作弊 efficiency 比 | 多为加性惩罚，显式效率比少见 | 🟡 方法论角度 |
| 小模型 ≤2.5B 文档导航 | 主流 7B+ | 🟡 角度 |
| snippet-only 观测 + 三层记忆 | 工程实现细节，非卖点 | — 工程而非 novelty |

## 4. 结构性 gap（新颖性候选区 —— 交给 idea-gen 锐化）

1. **文档内读取深度策略（doc-internal reading-depth）**：把 `read(section)` / `expand(neighbor)` 作为 RL 动作，控制"**单篇文档读多细**"，区别于 web search-depth。需要受控的 doc→section 语料。
2. **效率前沿（efficiency frontier）的严格刻画**：在受控文档导航环境上，系统比较"准确率-per-读取-token"前沿，而非单点 accuracy。
3. **反作弊效率比奖励**：`efficiency = gold-evidence tokens / total tokens read` 作为抗"全读再答"项，对照加性惩罚做消融。
4. **小模型（≤2.5B）文档导航**：验证效率前沿在小模型上是否更陡、是否更受益于 read-granularity 控制。
5. **"少用反而好"元认知**：把 §1-E 的警示（2602.07213）变成显式的"读前自评必要性"机制。

## 5. 已下载本地 PDF（papers/，共 16 篇）

`2210.03629`(ReAct) `2112.09332`(WebGPT) `2005.11401`(RAG) `2604.17337`(AutoSearch) `2602.22576`(Search-P1) `2510.15719`(Cost-Aware/Dynamic-Search-R1) `2503.09516`(Search-R1) `2503.19470`(ReSearch) `2503.05592`(R1-Searcher) `2505.14146`(s3) `2505.17391`(EVO-RAG) `2505.15117`(Empirical) `2510.16724`(Survey) `2506.04185`(R-Search) `2412.18547`(TALE) `2305.06983`(FLARE)。

## 6. 给 Phase 2 的硬约束

- 任何 idea 必须能回答：**"为什么不是 AutoSearch / Dynamic-Search-R1 的特例？"**
- 优先沿 §4 gap 锐化，避免在 §1-B 拥挤区重复。
- 全部 pilot 标注 needs manual pilot（仓库无数据集/GPU 配置）。
- 模型规模约束 ≤2.5B 必须保留。
