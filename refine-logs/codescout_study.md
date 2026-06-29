# CodeScout 精读报告 — 可迁移到 DocScout 的设计

> 论文：*CodeScout: An Effective Recipe for Reinforcement Learning of Code Search Agents* (Sutawika et al., arXiv:2603.17829, 2026-03)
> 参考实现：`reference/codescout/`
> 目标：把 CodeScout 在「代码定位」上的 RL 配方，提炼成 DocScout 在「NL 文档导航 + 读取预算」上可借鉴的设计。
> 每条结论后附论文页码 / 代码文件证据。

---

## A. 训练方式（RL 算法 + 超参 + reward 辅助项）

### A.1 算法：GSPO（Group Sequence Policy Optimization），不是 DR.GRPO

论文里 §3.4 写的是 GSPO（Zheng et al., 2025，arXiv:2507.18071）。**注意一个易混点**：论文称"Following DR.GRPO，我们去掉 KL 项和 advantage 的 std 归一化"——这是借用 DR.GRPO 的**两个具体 trick**，但主算法仍是 GSPO。代码证据：

- `scripts/run_async_training_1.7b.sh:89-92`：
  ```
  trainer.algorithm.policy_loss_type="gspo"
  trainer.algorithm.eps_clip_low=0.0003
  trainer.algorithm.eps_clip_high=0.0004
  trainer.algorithm.loss_reduction="sequence_mean"
  ```
- KL 关闭：`run_async_training_*.sh` 里 `use_kl_loss=False`、`use_kl_in_reward=False`（论文 §3.4，p.4）。
- std 归一化关闭：`trainer.algorithm.grpo_norm_by_std=false`。

**GSPO 的核心差异**（论文 Eq.2-3，p.4 + Appendix E，p.20）：
- 用 **sequence-level importance ratio**（对整条 sequence 的 token 对数似然比取几何平均，Eq.3），而非 token-level。
- **tight clipping**：`eps_clip_low=3e-4, eps_clip_high=4e-4`（0.0003/0.0004），比标准 PPO 的 0.2 窄 ~500 倍。
- loss reduction 用 `sequence_mean`。

### A.2 关键技巧

1. **去掉 KL 正则**（§3.4, p.4；代码 `use_kl_loss=False / use_kl_in_reward=False`）。
2. **去掉 advantage 的 std 归一化**：`Â_i = r_i − mean(r)`（Eq.4），**不除 std**（论文明确写 "remove the standard deviation from the advantage calculation"，§3.4 p.4）。
3. **disable entropy loss + mask 掉"耗尽步数不提交"的 rollout**：论文 §3.4 "We also disable entropy loss and mask loss for rollouts that exhaust maximum steps without calling the finish tool"。
4. **token-level loss masking**：对非模型生成 token（system/user prompt、tool observation）mask 掉 loss（§3.4 末，p.4-5）。
5. **sequence-extension trick**：对 1.7B/14B 关 thinking，但改 chat template 保留历史 `<think>` token，保证"前缀可拼接"，把多步 trajectory 合并成单条 sequence 训练（§3.4-p5 顶部 + 脚注1）。

### A.3 `r_turn` 辅助项 —— 防"耗尽步数不提交"（论文 §3.3，p.4）

这是 DocScout 最该抄的一点。论文原文（p.4）：

> "In early experiments for CodeScout-14B, we observed training collapse characterized by near-zero rewards in later stages... the agent frequently exhausted the step budget without submitting predictions. To mitigate this, we use an auxiliary binary reward `r_turn(τ, k)` that assigns 1 **if and only if** the agent terminates in exactly k turns, where k is the step limit."

代码实现（`src/rewards/multiturn.py`）：
```python
def multiturn_reward(messages, maximal_turns=5, minimal_turns=1, **kwargs):
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    num_turns = len(token_messages)
    if (num_turns >= minimal_turns) and (num_turns <= maximal_turns):
        return 1.0
    return 0.0
```
- **只在 14B config 启用**：`configs/reward_config_14b.yaml` 里 `multiturn_reward` 的 `minimal_turns=4, maximal_turns=4`（即"恰好在第 4 步＝步数上限提交"才给 +1）。
- **4B 和 1.7B 不用**（`reward_config_4b.yaml` / `reward_config_1.7b.yaml` 只有 `multilevel_localization_f1_reward`）。
- 论文还提了另一个被否的辅助项：**并行 tool-calling 奖励会 hurt 性能**，所以改成"在 prompt 里显式要求并行调用"（§3.3 末，p.4）。

### A.4 训练步数 / 数据量 / 群组大小（论文 §4.1，p.5）

| 模型 | 起点 | 步数 | 训练实例数 | batch size | rollouts/instance | max context | max turns | r_turn(k) |
|---|---|---|---|---|---|---|---|---|
| CodeScout-4B | Qwen3-4B-Instruct-2507 (base, 直接 RL) | 200 | 1.6K | 8 | 8 | 40K | 6 | 无 |
| CodeScout-14B | Qwen3-14B (base, 直接 RL) | 300 | 9.6K | 32 | 4 | 50K (YaRN×2) | 4 | k=4 |
| CodeScout-1.7B-RFT | Qwen3-1.7B → RFT → RL | RFT:1epoch/4K | RFT:7.7K采→4K成功 | RFT:8 | — | 32K | 4 | 无 |
| CodeScout-1.7B (RL阶段) | 1.7B-RFT checkpoint | 100 | 800 (未在RFT见过) | 8 | 8 | 32K | 4 | 无 |

- **总训练池：39K instances，跨 128 个 repo**（§4.1, p.5），由 SWE-Smith (Yang et al., 2025b) 经 §3.1 流程处理。
- **统一超参**（§4.1 末，p.5）：lr=1e-6（constant，不衰减）、AdamW、clip low=3e-4 / high=4e-4、rollout temperature=1.0。8×H100。
- **异步训练**：SkyRL 后端 + staleness≤4，rollout 与 weight 更新并行；每个 opt step 后同步 vLLM 权重并 kill 进行中的推理请求（§3.4，p.4）。
- **RFT 细节**（§4.1，p.5）：从 CodeScout-14B 在 7.7K 实例上采样，只留"三个粒度 F1 全=1.0"的轨迹，得 4K；用 veRL 框架 SFT 1 epoch，lr=5e-5 cosine，warmup 0.1，global batch 8。**RFT loss→0 后再做 RL 仍能继续涨**（Appendix B，p.17-18），前 ~20 步陡升后饱和。
- **解码**（评测时，§4.2, p.6）：thinking off 用 temp=0.7, top-k=20, top-p=0.8；thinking on 用 temp=0.6, top-p=0.95；max ctx 132K。

### A.5 算法消融的关键发现（Appendix E，p.20-21，Table 9）

在 Qwen3-4B 上 200 步、SWE-Bench Pro 评测：
- GSPO/SAPO/Dr.GRPO/GRPO 四种 critic-free 配方，file-F1 都在 47-55%，**任务对 RL 算法选择不敏感**（"the choice of reward design and agent scaffold likely matters more than the specific RL algorithm"，p.21）。
- **把 seq-mean 改成 length-unbiased token-sum 会让 GSPO file-F1 暴跌 54.83→42.02**（p.21，Table 9 下半）——说明 sequence-level ratio 不兼容 Dr.GRPO 式归一。
- 开 std 归一化 file-F1 略降（54.83→52.73）但 module/function-F1 升（31.43→34.37, 23.29→26.41）。

---

## B. 数据 / 环境设计思想

### B.1 任务定义：三粒度 code localization（论文 §3.1，p.3 + Figure 2）

给定 GitHub issue `I` + pre-PR 仓库 `R`，从 gold patch `P` 提取 ground truth `y* = (F*, M*, U*)`：
- `F*` = 被修改的文件集
- `M*` = 被修改的 module（类）集
- `U*` = 被修改的函数/方法集

提取脚本借用 LocAgent 的 patch-processing，并增强：(i) 检测 module/file 级成员函数与类属性新增；(ii) 捕获 import 语句与全局变量改动；(iii) 忽略函数/类内 docstring 改动（§3.1，p.3）。

**三粒度 F1 同时计算并相加作为 reward**——这正是其 reward 信号丰富、能学到细粒度行为的关键（见 C）。

### B.2 为什么用最小工具集（仅 1 个 terminal）（论文 §2 + §3.2，p.2-3）

- **语言无关**：bash terminal 天然不依赖任何语言的静态分析（AST/call graph）。对比 LocAgent/CoSIL/RepoSearcher/RepoNavigator 都要 Python 专用解析器（Table 1, p.2；Table 8, p.19）。
- **工具数 1 vs 3-5**（Table 1, p.2）：action space 小，RL 探索成本低。
- **环境开销极低**（§3.1 末, p.3）：只需 git clone pre-PR commit 到固定路径，**不装依赖、不沙箱、不容器化**——因为"localization 不需要执行代码"。这对 RL rollout 大规模并发至关重要。
- 命令收敛现象（§6.2, p.10-11, Figure 3）：训练后 CodeScout-14B 只用 `rg` + `sed` 两个命令，4B 用 `rg/sed/cat/find/xargs`。"有效定位用极小子集 Unix 工具即可"。

### B.3 可解性保证（necessity / sufficiency）（论文 §3.1 + §4.1）

- **过滤掉 PR 创建/删除文件的 issue**（agent 无法预测新文件名，删文件无法定 module/function）（§3.1, p.3）。
- **忽略非 Python 文件**（如 README.md）作为 ground truth——因为提不出 module/function（§3.1, p.3）。
- **丢弃空 issue description**（§3.1, p.3；代码 `build_dataset.py:22`）。
- **no-repo-overlap**：39K 训练实例来自 SWE-Smith，**128 个 repo 全部不与评测 benchmark 重叠**，避免污染（§4.1, p.4-5）。
- **gold 来自 patch 的"被修改区域"**——这是必要性近似（"must modify to fix"），但有局限：可能漏掉"理解 issue 需要读但不需要改"的代码（Appendix D, p.20，论文自承 limitation）。
- **RepoNavigator 的对照做法**（§C.1, p.19）：丢弃"base Qwen2.5-7B 在 16 次采样里一次都没解出"的实例来选 easy 子集——CodeScout 不这么做，直接全量 RL。

### B.4 难度分档

论文**没有显式的难度分档**用于训练；难度由 benchmark 体现：
- SWE-Bench Lite (300) < Verified (500) < Pro Python 子集 (266，"substantially more challenging"，§4.2, p.5-6)。
- 隐式分档体现在"1.7B 太弱需 RFT warm-start"（base Qwen3-1.7B 在评测上 near-zero F1，§4.1, p.5）——即模型能力档决定是否需要 SFT 暖身。

### B.5 可迁移到 NL 文档导航的思想

| CodeScout 概念 | DocScout 对应 |
|---|---|
| 三粒度 gold (file/module/function) | section 树（doc/section/paragraph），gold 证据可定位到段落级 |
| `rg` 关键词搜索 | `search(query)` |
| `sed -n 'a,bp'` 读行范围 | `read(doc, section)` / `expand(section)` |
| `localization_finish` 结构化提交 | `answer(...)` 带证据引用 |
| pre-PR commit clone（不需执行） | 预切好的 doc 语料（不需运行） |
| patch 提 gold（necessity） | MuSiQue/2WikiMultihop 的 gold supporting facts（necessity + sufficiency 已标注） |
| "gold 来自被修改区域" 的局限 | DocScout 应同时纳入"必读但不必引"的证据——靠 evidence-F1 over committed reads 缓解 |

**关键迁移点**：CodeScout 的"环境零开销"思想（不执行、不沙箱）对 DocScout 天然成立——NL 文档导航本来就不需要执行，rollout 可大规模并发。

---

## C. Benchmark 测评内容

### C.1 指标（论文 §4.2，p.6）

- **主指标**：instance-wise 平均 F1，三粒度（file / module / function）分别报。F1 = precision 与 recall 的调和平均。
- 同时报 **precision 和 recall**（§4.2, p.6）——因为很多 baseline 预测固定 top-K（K=5），recall 高 precision 低；CodeScout 动态预测可变数量位置，precision 更重要（§4.3, p.7 引 Pan et al.："污染 context 比漏掉 context 更有害"）。
- 代码里 F1 计算在 `src/rewards/file_localization/file_localization.py:7-14`（`compute_file_f1_score`，带 beta 参数，默认 1.0）。

### C.2 效率维度（论文 §6.1 + §6.2，p.10；Table 6）

**有明确的效率评测**。Table 6（issue resolution + localization 联合实验，p.10）报：
- Resolution Rate ↑
- **Avg. # Steps ↓**
- **Avg. Input Tokens ↓**
- **Avg. Output Tokens ↓**

发现：用 CodeScout-14B 的定位结果增强 issue-resolution agent，**步数、token 都下降**（Qwen3-4B：步数 16.09→13.91，input token -17.46%，output -6.71%；30B：步数 -1.37，input -10.12%）。代码侧效率度量在 `src/metrics/efficiency_metrics.py`（tokens/steps/avg_tool_calls_per_step/wall_clock）和 `src/metrics/trajectory_metrics.py`（num_turns/num_tool_calls/num_tool_calls_per_turn/parallel grouping）。

**但注意**：CodeScout 训练 reward 里**没有效率项**（reward 只是三粒度 F1 之和 + 14B 的 r_turn）。效率只在"下游 issue resolution"实验里作为评测维度，且定位本身用 max_turns 硬截断（4 或 6）而非软惩罚。这正是 DocScout 的创新空间（见 D）。

### C.3 小模型(1.7B)怎么打败 14B（论文 §5.1，p.7-8）

核心机制 = **RFT warm-start + RL**：
1. base Qwen3-1.7B 直接 RL 不行（near-zero F1，无有效梯度）。
2. 先从 CodeScout-14B 在 7.7K 实例采样，留"三粒度全 F1=1.0"的 4K 成功轨迹做 RFT（蒸馏），得 CodeScout-1.7B-RFT（file-F1 46.6 on Verified）。
3. 再 RL 100 步 → CodeScout-1.7B（file-F1 55.46，**超过 Qwen3-14B base 的 43.13 达 12 个点**，§5.1 p.7）。

**关键数据**（Table 3, SWE-Bench Verified, p.6）：
- CodeScout-1.7B (55.46 file-F1) vs Qwen3-14B base (43.13)：**1.7B 反超 8× 大模型 11-18%**。
- vs Qwen3-32B-Thinking (62.91)：file 落后 2-7%，但 module/function 反超 2-6%。

其他发现：
- **结构化 finish 工具 > 字符串解析**（§3.2, p.3）：早期用 Chen et al. 的 string 输出格式，reward 信号因解析脆弱而噪声大；改用 `localization_finish` 强制结构化 schema（`src/tools/localization_finish.py`）后显著改善 reward 保真度。
- **前沿闭源模型对 prompt 极度敏感**（§5.3, p.8-9）：GPT-5/Claude-Sonnet-4.5 不加"最后一轮提交提醒"（`OpenHands-Bashrem`）时几乎全 0 分（耗尽步数不提交）；加提醒后暴涨（GPT-5 file-F1 3.2→78.18）。这反衬出 r_turn 辅助项对小模型 RL 的价值。

---

## D. 对 DocScout 的具体迁移建议

DocScout 现有 reward（`docscout/reward/reward.py`）已借鉴了 r_turn 思想（`r_submit_bonus` / `r_nosubmit_penalty`）。以下是对照 CodeScout 后的进一步建议。

### D.1 应照搬的设计

1. **多粒度 F1 reward（最重要）**。CodeScout 的 reward = `F1_file + F1_module + F1_func`（Eq.1, p.4；代码 `multilevel_localization_f1_reward`）。DocScout 当前 evidence 是单一 section-F1（`evidence_f1_read`），**建议升级为多粒度**：`F1_doc + F1_section + F1_paragraph`，让 reward 信号更稠密、能学到"先定位文档再缩到段落"的层次行为。这与 DocScout 的 section 树天然契合。

2. **结构化 finish 工具**。CodeScout 用 pydantic schema 的 `LocalizationFinishAction`（`src/tools/localization_finish.py`）替代字符串解析，直接消除 reward 噪声。DocScout 的 `answer(...)` 工具应同样强制结构化（answer 字段 + evidence list 字段），避免 citation 解析脆弱（reward.py 注释里已意识到 "brittle citation parsing"）。

3. **去掉 KL + 去掉 advantage std 归一化 + mask 未生成 token + mask "耗尽步数不提交"的 rollout**（§3.4, p.4）。这些都是 zero-cost 的稳定化 trick，DocScout 的 GRPO 实现应直接照抄：
   - `use_kl_loss=False / use_kl_in_reward=False`
   - advantage = `r_i − mean(r)`（不除 std）
   - 对耗尽步数的 rollout mask loss（而非给负 reward——mask 比 penalty 更干净，避免 hack）。

4. **超参起步值**：lr=1e-6 constant、clip=3e-4/4e-4、rollout temp=1.0、batch 8、rollouts/instance 8（§4.1, p.5）。这些是 CodeScout 在 1.7B-14B 上验证过的稳健默认值。

5. **小模型 RFT→RL 两段式**。若 DocScout 用 Qwen3-1.7B 且 base 在 MuSiQue 上 near-zero，**不要直接 RL**（无有效梯度）。先从更大模型（或 GPT）采样"全 gold 命中"轨迹做 RFT，再 RL。CodeScout 证明 RFT loss→0 后 RL 仍能继续涨（Appendix B, p.17-18）。

6. **no-data-overlap + 过滤不可解实例**。DocScout 训练集的文档/QA 必须与 MuSiQue/HotpotQA 评测集**零文档重叠**；过滤掉 gold evidence 为空或 question 为空的实例（对应 CodeScout 过滤空 issue / 创建删除文件的 PR）。

### D.2 read-budget 奖励能从 CodeScout reward 借鉴什么（核心问题）

CodeScout 的 reward **本身没有效率项**——它的"效率"完全靠 (a) max_turns 硬截断 + (b) r_turn 二值终止奖励 来隐式约束。这给 DocScout 两个对照启示：

1. **r_turn 的二值化思想可用于 DocScout 的终止塑形**。CodeScout 的 `multiturn_reward` 是"恰好第 k 步提交才 +1"的**硬二值**（`minimal_turns=maximal_turns=k=4`）。DocScout 当前的 `r_submit_bonus=0.5 / r_nosubmit_penalty=-0.5` 是"提交即 +0.5"的**软二值**。建议：可做一个 ablation，试 CodeScout 式的"恰好在预算步数提交才给 bonus"（更激进地逼 agent 用满预算再提交），vs 当前的"只要提交就给 bonus"。注意 CodeScout 只在 14B（易 collapse）启用 r_turn，1.7B/4B 不用——**暗示小模型不需要终止塑形，大模型才需要**。DocScout 若用 1.7B 可能可以去掉终止塑形简化 reward。

2. **把 CodeScout 的"多粒度 F1 之和"作为 DocScout reward 的 accuracy 主干**，再**叠加** DocScout 独有的 efficiency-ratio 项。即：
   ```
   R = (F1_doc + F1_section + F1_paragraph)  ← 抄 CodeScout 的 accuracy 主干
       - γ·(1 - efficiency_ratio)            ← DocScout 独有的 read-budget 项
       + r_turn (仅大模型)                    ← 抄 CodeScout 的终止塑形
   ```
   CodeScout 证明了"纯 F1 reward + 结构化工具 + GSPO"就够把 1.7B 训到反超 14B；DocScout 的效率项是**在这个已验证主干上的增量**。建议实验顺序：先复现"纯多粒度 F1 reward"（验证主干能 work），再加 efficiency-ratio（验证效率项不伤 accuracy 且改善 accuracy-per-read-token 前沿）。这符合 `refine-logs/自动化实验迭代方案.md` 的"先证环境可解、再 RL"方法论。

3. **"mask 而非惩罚"原则延伸到 read-budget**。CodeScout 对"耗尽步数不提交"是 **mask loss**（不给负 reward）。类比：DocScout 对"读了非 gold 内容"不应简单线性罚 token（`lambda_tokens * read_tokens` 易导致 agent 啥也不读），而应像 efficiency-ratio 那样**只在 committed read 存在时才生效**（reward.py 已这样实现：`ratio_cost ... if committed_read_tokens > 0 else 0.0`）——这与 CodeScout "不提交就 mask" 的哲学一致：**不奖励 hack，但也别创造新的 hack 面**。

4. **并行调用奖励会伤性能 → 别加**（§3.3, p.4）。DocScout 不要给"一次 search 多个 query"或"批量 read"加额外 reward，改为在 prompt 里显式鼓励。CodeScout 已踩过这个坑。

### D.3 评测协议借鉴

- **同时报 precision/recall/F1**（§4.2, p.6）：DocScout 评测 evidence 时也应拆 precision/recall，证明"高 precision 比高 recall 更重要"（对应 CodeScout 引 Pan et al. 的 context-pollution 论点，p.7）。
- **效率评测维度照搬 Table 6**：steps、input tokens、output tokens 三列。DocScout 的 accuracy-per-read-token 前沿应补充"avg steps"和"avg tokens"作为辅助效率指标，与 CodeScout 可比。
- **Reward/loss 曲线**（Appendix B）：训练时分别画 file/module/function-F1 随 step 变化。DocScout 应画 answer-F1 / evidence-F1(各粒度) / efficiency-ratio 随 step 变化，便于诊断是哪一项在涨。

---

## 可执行借鉴清单（按优先级）

| # | 动作 | 来源证据 | DocScout 落点 |
|---|---|---|---|
| 1 | reward 升级为**多粒度 F1 之和**（doc+section+paragraph） | 论文 Eq.1 p.4；`multilevel_localization_f1_reward` | `docscout/reward/reward.py` 增 `evidence_f1_multigranular` |
| 2 | **结构化 answer 工具**（pydantic schema，杜绝解析噪声） | 论文 §3.2 p.3；`src/tools/localization_finish.py` | DocScout `answer` 工具强制 schema |
| 3 | 去掉 KL、去掉 advantage std、mask 未生成 token、**mask 耗尽步数 rollout** | 论文 §3.4 p.4；脚本 `use_kl_loss=False` 等 | DocScout GRPO 训练循环 |
| 4 | 超参默认：lr=1e-6、clip=3e-4/4e-4、temp=1.0、batch8×rollout8、staleness≤4 | 论文 §4.1 p.5；`run_async_training_1.7b.sh` | `configs/` 训练配置 |
| 5 | 小模型走 **RFT(全 gold 命中轨迹)→RL** 两段式 | 论文 §4.1 p.5；Appendix B p.17 | 1.7B 若 base near-zero 则先 SFT |
| 6 | 终止塑形：大模型用 r_turn 硬二值（恰在第 k 步提交），小模型可省 | 论文 §3.3 p.4；`multiturn.py`；仅 14B config 启用 | ablation `r_submit_bonus` 形态 |
| 7 | efficiency 项只在 committed read>0 时生效（不创造 hack 面） | 对照 CodeScout "mask 而非惩罚" 原则 | reward.py `ratio_cost` 已如此，保持 |
| 8 | **不加**并行 tool-calling 奖励（会 hurt），改为 prompt 鼓励 | 论文 §3.3 p.4 | DocScout search/read prompt |
| 9 | 评测报 precision/recall/F1 + steps/tokens 双维度 | 论文 §4.2 p.6 + Table 6 p.10 | DocScout eval 脚本 |
| 10 | 训练曲线分项画（answer/evidence 各粒度/efficiency） | 论文 Appendix B p.17 | wandb logging |
| 11 | no-doc-overlap + 过滤空 gold/空 question 实例 | 论文 §3.1 p.3 + §4.1 p.5 | DocScout 数据构造 |
| 12 | 实验顺序：先证"纯多粒度 F1 reward"主干 work，再加 efficiency-ratio | 论文 reward 主干已验证 + DocScout 增量创新 | `自动化实验迭代方案.md` |

---

### 证据索引（关键文件路径）
- 论文：`related-papers/2603.17829.pdf`（A=§3.4/§4.1/App E；B=§3.1/§3.2；C=§4.2/§5/§6/Table 3,6,9；reward=§3.3 Eq.1）
- reward 核心：`reference/codescout/src/rewards/file_localization/file_localization.py`（`multilevel_localization_f1_reward`, `compute_file_f1_score`）
- r_turn：`reference/codescout/src/rewards/multiturn.py`
- reward configs：`reference/codescout/configs/reward_config_{1.7b,4b,14b}.yaml`（仅 14b 含 multiturn）
- 结构化 finish：`reference/codescout/src/tools/localization_finish.py`
- 训练超参：`reference/codescout/scripts/run_async_training_{1.7b,4b,14B}.sh`（GSPO/clip/KL/lr/temp/max_turns）
- 数据构造：`reference/codescout/src/build_dataset.py`（过滤空 problem_statement）
- 效率度量：`reference/codescout/src/metrics/{efficiency_metrics,trajectory_metrics}.py`
- agent/generator：`reference/codescout/src/agent/agent.py`、`src/generator/code_search_generator.py`
- DocScout 现状（已借鉴 r_turn）：`docscout/reward/reward.py`
