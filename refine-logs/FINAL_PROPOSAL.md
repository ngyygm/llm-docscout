# FINAL PROPOSAL — DocScout（NL 文档版的 CodeScout，读取预算一等）

> Phase 4.5 输出。已吸收 Phase 4 评审的 reframe（从"迁移"→"finding+method"）。

## Problem Anchor（冻结，防漂移）
> 给定自然语言文档语料库 + 问题，训练 ≤2.5B RL 策略，用最小工具集 **{search, read(section), expand(neighbor), answer}** 定位并抽取正确答案，**在"读取预算"（读入上下文的 token）约束下最大化准确率**，并刻画 **accuracy-per-read-token 前沿**。
> 锚点参照：CodeScout(2603.17829) 证明了"最小工具 + GRPO + 数据/环境工程"在 **code** 上让 1.7B 打败 14B。本工作问：**在 NL 文档上，这套范式 + 读取预算奖励会得到什么？**

## Method Thesis（一句话，已 reframe）
> **在长 NL 文档上，"文档内读取"是主导成本；read-budget-aware 策略在 accuracy-per-read-token 前沿上支配固定预算 RAG 与 search-depth-RL；efficiency-ratio reward 比加性/冗余惩罚更优且抗"全读再答"作弊。**

## Dominant Contribution（4 件套）
1. **受控 NL 文档导航环境**：doc→section 树，section 可定位 gold 证据；合成语料（可控长度/密度）+ 真实 benchmark（MuSiQue/HotpotQA 段落作 section）。
2. **read-budget reward 设计**：答案正确性(EM/语义) + 证据 section-F1 + 三种成本项对照 {加性 −λ·read_token / ALDEN式冗余惩罚 / **efficiency-ratio(本工作)**}。
3. **accuracy-per-read-token 帕累托前沿**作主评测 + 明确测量协议（扫 λ / 扫预算上限）；次轴 #tool-calls、latency。
4. **实证 finding**：search-depth-RL(Search-R1/AutoSearch 适配) 与固定预算 RAG 被支配；小模型前沿分析。

## 方法细节

### 环境 / 动作 / 观测
- **动作**：`search(query)→top-k snippets`（BM25+dense 混合，**只返回 snippet 不返回全文**）；`read(doc,section)→完整 section + 邻居可用性`；`expand(doc,section,dir)→邻居 section`；`answer(text,evidence)→终止`。
- **成本量纲**：`tokens_read` = **真正读入上下文**的 token（区别于检索返回 token）。
- **观测（三层记忆）**：question（常驻）+ 紧凑 action_log + 当前候选 snippet + 小证据缓冲。

### 奖励三变体（核心消融对象）
```
R_add    = score_answer + α·section_F1(evidence) − λ·tokens_read
R_redund = score_answer + α·section_F1 − β·redundancy_penalty        # ALDEN 式
R_ratio  = score_answer + α·section_F1 − γ·(1 − efficiency_ratio)    # 本工作
          efficiency_ratio = gold_evidence_tokens / total_tokens_read   # 基于证据归因
```

### RL
- **DR.GRPO**（去 KL、去 advantage std，直接照搬 CodeScout §3.4），SkyRL 异步后端。
- 模型：Qwen3-**1.7B / 2.5B**（主），7B（对照，若算力允许）。
- 防"耗尽步数不提交"：沿用 CodeScout 的 `r_turn` 辅助二值项。

## 数据
- **合成**：自动生成多 section 长 NL 文档（政策/技术手册风格），植入 QA + section 可定位 gold 证据；可控长度×密度×多跳数。用于隔离消融。
- **真实**：MuSiQue / HotpotQA（段落→section，多跳）；评估泛化。Stage 2 落地时确认 section-locatable 证据可得性。

## 差异化（一句话，给审稿人）
> "CodeScout 学在**代码**里定位（accuracy-only，max_turns 截断）；DocScout 学在**NL 文档**里**按读取预算**定位+抽取，刻画 accuracy-per-read-token 前沿——与 DeepRead/IntrAgent(prompting)、ALDEN(VLM+防冗余)、AutoSearch(web 搜索深度) 均不同。"

## 剩余风险（承自评审）
1. ALDEN 是强竞品 → 必须做其 reward 变体对照（R_redund）并在 related work 精准切割。
2. 真实 benchmark 必须落地至少一个。
3. 前沿测量协议要写清，否则"前沿"被质疑不可复现。
4. 小模型 scaling claim 要么坐实要么删。
