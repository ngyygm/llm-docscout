# EXPERIMENT PLAN — DocScout（claim 驱动）

> Phase 4.5 输出。每个 claim ↔ 一组实验。pilot 全部标注 **needs manual pilot**（仓库当前无数据集/GPU 配置）。

## Claims ↔ Experiments

| Claim | 实验 | 关键基线/对照 | 主指标 | kill 条件 |
|---|---|---|---|---|
| **C1** read-budget 策略在前沿上支配固定预算 RAG & search-depth-RL | **E1** 训 DocScout(R_ratio) + 全基线，画前沿 | fixed top-k RAG(k 扫)、FLARE、Search-R1-style、AutoSearch-style | accuracy-per-read-token 帕累托 | DocScout 前沿不被基线支配即成立 |
| **C2** efficiency-ratio reward > 加性/冗余(iso-accuracy) | **E2** reward 变体消融 | R_add vs R_redund vs R_ratio | 同准确率下 read-token | R_ratio 无显著优 → 降为工程细节 |
| **C3** 文档内读取粒度(expand)在长文档上比搜索深度更关键 | **E3** 动作空间消融 × 文档长度档 | 去 expand / 只 search / 完整 | 前沿 + 按长度分层 | expand 无增益 → 删该动作 |
| **C4** 小模型前沿更陡/更受益 | **E4** 规模消融 | 1.7B / 2.5B / (7B) | 前沿斜率 | 趋势不显著 → 删 scaling claim |

## 必跑消融
- **A-reward**: {R_add, R_redund, R_ratio}（回应 W1/W2，切割 ALDEN）
- **A-action**: {full, −expand, −read(只search)}（回应 C3）
- **A-scale**: {1.7B, 2.5B, 7B-if-budget}（回应 C4）
- **A-doclen**: {短/中/长} × {单跳/多跳}（回应"长文档读取是主导成本"）
- **A-hack**: 紧预算下失败模式分析（回应 W5：不读就幻觉/读一片就猜）

## 基线清单（审稿人会查）
1. fixed top-k RAG（k 扫描 → 自带前沿）
2. FLARE(2305.06983) active retrieval
3. Search-R1-style(2503.09516) **适配到文档设置**（必做，否则答不硬"为何不是其特例"）
4. AutoSearch-style(2604.17337) **适配**（控搜索深度）
5. DeepRead/IntrAgent prompting 基线（若可复现）
6. ALDEN-text-adapt 或至少其 R_redund reward 变体（切割头号竞品）

## 主指标与协议
- **主**：accuracy-per-read-token 帕累托前沿。测量：扫 reward λ ∈ {…} 或扫预算上限 B，每点取 N=3 seed 均值，画前沿；报告前沿下 AUC 与 iso-read-token 准确率。
- **次**：#tool-calls、wall-clock latency、EM/F1、证据 section-F1。

## 第一批 run（Stage 2 启动时，需先配 GPU）
> 均标注 **needs manual pilot**：
1. **R0 环境冒烟**：搭合成 env + reward 计算 + GRPO 桩；用 Qwen3-1.7B **base（不训练）** 跑 rollout，验证 env/reward/section 归因正确。
2. **R1 训练冒烟**：DocScout-1.7B + R_add，合成小批(≈2K 实例)，确认 reward 上升、不崩。
3. **R2 首个前沿**：R_add vs R_ratio 在合成集上的 accuracy-per-read-token 前沿（C2 初判）。

## Stage 2 实现参考（用户已提供）
- 克隆参考实现：`git clone https://github.com/OpenHands/codescout`
- 模型/集合：`https://huggingface.co/collections/OpenHands/codescout`
- 重点参照：**SkyRL 后端接入**、**DR.GRPO 配置**、**reward 计算脚本**、**OpenHands-Bash 脚手架 → 改造为 NL 的 search/read/expand**、**数据/环境 curate 脚本**。

## 算力/数据备忘
- GPU：用户有（具体配置 Stage 2 需接入）。
- 数据：无 → Stage 2 先合成，再接 MuSiQue/HotpotQA。
- 时间盒：pilot 估 ≤2h/GPU（PILOT_MAX_HOURS）；首批 3 run 估 < 8 GPU·h。
