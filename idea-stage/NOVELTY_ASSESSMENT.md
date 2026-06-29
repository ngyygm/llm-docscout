# 新颖性评估 — Phase 3

> 锚点论文（用户指定）：**CodeScout (arXiv 2603.17829)** — RL 训练的**代码定位** agent，纯 Unix 终端工具，DR.GRPO，三粒度(file/module/func) F1 奖励，Qwen3-1.7B/4B/14B。
> 用户方向：**做 CodeScout 范式的自然语言文档版本**。

## 1. CodeScout 精确拆解（移植 NL 的基线参照）

| 维度 | CodeScout 做法 |
|---|---|
| 任务 | 代码定位：issue+repo → 预测被 patch 改动的 (文件, 模块, 函数) |
| 工具 | OpenHands-Bash = 纯 Unix 终端（rg/grep/find/cat…），max_turns 4–6，并行工具调用 |
| 奖励 | `r = F1_file + F1_module + F1_func`；14B 加辅助 `r_turn`（恰好 k 步终止=1，防"耗尽步数不提交") |
| RL | DR.GRPO（去 KL、去 advantage std），SkyRL 异步后端 |
| 数据 | SWE-Smith → 39K 实例/128 仓库；评测 SWE-Bench Verified/Pro/Lite |
| 模型 | Qwen3-1.7B/4B/14B（1.7B 已打败 8× 的 Qwen3-14B） |
| 效率观 | **只优化定位准确率；max_turns 硬截断；辅助 reward 仅为"别拖"——不优化读取量** |

## 2. 核心洞察：code→NL 不是平移，而是换难题类型

- **code 有精确结构锚点**：grep 命中即精确；文件/模块/函数层级确定性；答案=可枚举位置集合，precision/recall 干净。
- **NL 无此锚点**：证据被改写（grep 找不到）、section 边界软、答案常是自由 span。
- → NL 版需要：**语义 search + section-aware read/expand**，且奖励不能是纯定位 F1，需 **答案正确性 + 证据归因**。
- → **读取预算/效率**在 NL 更关键（无锚点→更容易过度读取污染上下文），而 CodeScout 完全没把它当一等目标。**这就是护城河。**

## 3. 威胁矩阵（已用 deepxiv 验证训练方式）

| 论文 | 领域 | 工具 | 训练 | 效率目标 | 与本工作差异 |
|---|---|---|---|---|---|
| **CodeScout 2603.17829** | code | terminal grep | **RL(GRPO)** | ❌(硬截断) | 锚点；本工作=NL + 读取预算一等 |
| DeepRead 2602.05014 | 文档(PDF) | Retrieve+ReadSection | **prompting**(无RL) | ❌ | 结构感知但非 RL；本工作=RL 配方 |
| IntrAgent 2604.22861 | 文献 | section 排序精读 | **prompting**(无RL) | ❌ | 充分性检查但非 RL |
| **ALDEN 2510.25668** | **VLM 视觉富文档** | semantic+fetch(页索引) | **RL** | 🟡防冗余 | **最接近竞品**：但 VLM≠纯文本；防冗余≠读取预算前沿；无最小工具配方 |
| AutoSearch 2604.17337 | web QA | search | RL | 🟡搜索深度 | 控搜索深度，非文档内读取粒度 |
| Dynamic-Search-R1 2510.15719 | web QA | search | RL | 🟡cost-aware | 成本=检索 token，非读入上下文 token |
| SRAS 2601.01785 | 文档 | 文档选择 | RL(PPO) | ❌ | 文档级选择，非 section 读取粒度 |

## 4. 可辩护的新颖性位置（一句话）

> **"DocScout：把 CodeScout 的 RL 配方（最小工具集 + GRPO + 数据/环境工程）迁移到自然语言文本文档，并把'读取预算/效率'提为一等奖励目标——在受控 NL 文档导航环境上刻画 accuracy-per-read-token 帕累托前沿。"**

差异化三柱：
1. **领域+训练**：纯文本 NL 文档 × **RL 训练**（DeepRead/IntrAgent 是 prompting；ALDEN 是 VLM）。
2. **效率一等**：read-budget/efficiency 作为奖励主项 + 帕累托前沿评测（CodeScout 只看准确率、max_turns 硬截断；ALDEN 仅防冗余）。
3. **小模型配方**：≤2.5B（继承 CodeScout-1.7B 的参数效率证据，验证其在 NL 是否成立）。

## 5. Kill 条件核查

- ❓ 是否存在"纯文本 NL 文档 + RL + 读取预算 reward"的论文？→ 6 组定向 deepxiv 检索**未命中**；最接近的是 VLM 的 ALDEN。**判定：位置成立，但 ALDEN 的存在说明该空间活跃，必须在论文里明确切割。**
- ❓ 是否与 CodeScout 同质？→ 否（code vs NL；accuracy-only vs read-budget）。

## 6. 剩余风险（写入 Gate 1，供用户拍板）

1. ALDEN 是强竞品，需在 related work 里精准切割（VLM/视觉锚定 vs 纯文本/最小工具）。
2. "NL 文档导航 RL"空间活跃（DeepRead/IntrAgent/ALDEN 都近），novelty 强度=**中等偏上**，需靠"读取预算前沿 + 小模型 + 最小工具 RL 配方"组合撑住，单点都不够。
3. Phase 4 外部 GPT-5.4 对抗评审仍待 Codex 重新认证（当前为 Claude 自评）。
