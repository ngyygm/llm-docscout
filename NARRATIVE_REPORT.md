# Narrative Report — DocScout (AAAI target, 实测修订版)

> 喂给 `/paper-writing venue:AAAI`。基于 refine-logs/round-result.md 第 5–10 轮真实结果。
> 模型：Qwen3-1.7B + LoRA r=32（纠正：早期 frontier_positive.json 误标 7B，实测为 1.7B）。
> 硬件：本地 3× RTX 3090。

## 1. 问题与最初主张
标准 RAG 一次性检索 top-k 塞入上下文，浪费 token 在无关段落。我们想研究：能否用 RL 训一个带 read-budget 奖励的 search/read/expand/answer agent，在 **accuracy-per-read-token 前沿**上支配固定预算 RAG？最初主张（FINAL_PROPOSAL）：read-budget 策略（efficiency-ratio reward）支配固定 RAG，且 RL 修复 SFT 的停止行为。

## 2. 方法（已实现，可复现）
- **环境**：doc→section 语料；工具 search(BM25)→snippet、read(section)→全文、expand(neighbor)、answer→终止。三层紧凑记忆（question + action_log + 最近 N 读节）。
- **奖励三变体**：R_add=answer+w·evidence−λ·read_tokens；R_redund（ALDEN 式冗余惩罚）；R_ratio=answer+w·evidence−γ·(1−efficiency_ratio)，efficiency_ratio=committed_gold_tokens/committed_read_tokens（本工作）。+ 终止塑形（submit±0.5）。Round 10 加 action_cost（每读动作成本，罚浪费的重复读）。
- **训练**：SFT（answer 轮 4× 上采样修复"读了不答"病态）→ REINFORCE-with-baseline（跨实例 EMA baseline，克服 GRPO 在 peaked SFT 上的惰性）。
- **诚实度量**：committed read 用真实 BPE token（早期代码误用词数，已纠正 BPE/词≈1.14）。

## 3. 数据（难度逐级）
- **synth-v3（真实尺度，主底座）**：24 文档×10 节，**节均 129 词**（早期 toy 仅 26 词 → RAG 饱和）。gold 数值嵌在 intro 之后（**不在前 40 词 snippet 里** → RAG-snippet 去饱和至 0.18）。节内含 confounder（"标准层90天/审计365天/缓存7天"）考精读。train seed21 / eval seed77（值放置独立，无答案泄漏）。
- **MuSiQue**（真实多跳，泛化测试）：18 文档/实例，段均 101 词，2–4 跳。

## 4. 关键结果（诚实，含负面）

### 4.1 诊断先于训练（方法论价值，正面）
- 审计发现早期"正向"结果三个诚信问题：①"token"实为词数；②模型误标 7B（实测 1.7B）；③toy 底座上 agent 输给 RAG（0.73 vs 0.96）。**诊断先于训练**抓出了底座不足。

### 4.2 真实底座前沿（synth-v3, n=150）
| 方法 | acc | read BPE | 说明 |
|---|---|---|---|
| oracle（gold 全文） | 0.827 | 239 | 可解上限（非饱和） |
| rag_full k=1 | 0.753 | 182 | 公平强 RAG |
| rag_full k=3 | 0.813 | 546 | |
| rag_full k=5 | 0.820 | 907 | 边际递减 |
| rag_snip k=5 | 0.180 | 279 | snippet-only 去饱和 |
| **agent（SFT≈RL-ratio≈RL-add）** | **~0.65** | **~160** | **被 rag_full_k1 支配** |

### 4.3 核心负面发现 + 机制（论文主贡献）
1. **read-budget RL 惰性**：RL-ratio=RL-add=SFT（单跳/多跳逐实例完全相同）。因 SFT 示范"读gold→答"已达 max reward 1.8，RL 无杠杆。
2. **瓶颈是选择不是读取**：失败分类显示 agent 31% 没读到 gold（selection_fail）> 19% 读到 gold 但答错（read_fail）。读到 gold 时 acc=0.718（接近 oracle）。**read-budget reward 管不到"读哪个候选"**。
3. **agent 选择比 BM25 top-1 还差**：rag_full_k1 直接读检索 top-1（含 gold ~75%），agent 学到"读1-2就答"漏掉深位 gold → 准确率被选择卡死 → 被 RAG-k1 支配。
4. **adaptive 示范未修好选择**：按 rank 顺序读直到覆盖 gold 的示范，acc 反降至 0.573（agent 学众数"读1-2"，多读引 confounder 干扰）。

### 4.4 read-budget RL 的真实生效前提（take-away）
需同时满足：(a) 选择非瓶颈（检索强或 agent 选得准）、(b) 任务有"读多少"的真实自由（需读多节且解不唯一）。synth-v3 单跳主导，两者皆不满足。

## 5. 图表清单
- T1：synth-v3 前沿表（上表，核心）。
- F1：accuracy-per-read-token 帕累托前沿图（agent 簇 vs RAG-k1/3/5 vs oracle）—— 显示 agent 被 RAG-k1 支配。
- F2：失败分类饼图（selection 31% vs read 19% vs correct）。
- F3：RL 惰性证据（RL 与 SFT 逐实例 acc 散点，完全对角）。
- T2：Phase-1 可解性（Recall@5=0.847，oracle 0.827）。

## 6. 诚实局限
- read-budget 论点（agent 支配 RAG）**未被证实**；反而证伪于真实底座（被 RAG-k1 支配）。论文如实报告 + 给出生效前提。
- MuSiQue 零样本迁移失败（v3-SFT 0.04）——synth 策略不迁移到真实多跳，需在 MuSiQue 上训练（未来工作）。
- 单种子、n=150（统计功效有限；RL≈SFT 的结论由"逐实例完全相同"强支撑，非依赖 p 值）。
- 1.7B 能力有限；7B/更大模型能否改变结论未测（Qwen2.5-7B 已缓存，未来工作）。

## 7. 论文定位
**诚实诊断研究**：不是"read-budget RL work 了"，而是"用严谨方法论刻画 read-budget RL 在 NL 文档 QA 上**何时不 work 及为何**"。贡献 = 诊断方法论 + 真实底座 + 清晰的失效机制 + 生效前提。负面但有价值（多数 agentic-RL 论文只报正面，仔细的负面+机制同样重要）。
