# DocScout 逐轮迭代结果（round-result.md）

> 遵循 `refine-logs/自动化实验迭代方案.md` 的「最推荐实际开工路线（§十二）」逐轮推进。
> 每轮记录：**做了什么 / 解决了什么 / 还有什么要探究**。每轮末由一个独立子 agent 判定是否完成（未完成不进入下一轮）。
> 硬件：本地 3× RTX 3090（0=3090Ti）。模型主线：Qwen3-1.7B（≤2.5B 符合 proposal）。

---

## 状态总览（截至 2026-06-27）

| 轮次 | 主题 | 状态 | 底座 | 结论 |
|---|---|---|---|---|
| 第一轮 | 验证环境可解性 | ✅ 完成 | toy-synth | Recall@5=0.847；oracle synth 0.85 可解 |
| 第二轮 | 无训练基线 | ✅ 完成 | toy-synth | oracle 0.748 / RAG 0.609 / prompted 0.110 |
| 第三轮 | 示范轨迹 + SFT | ✅ 完成 | toy-synth | 发现并修复 SFT「读了不答」退化（answer 轮 4× 上采样） |
| 第四轮 | 最简 RL（GRPO/REINFORCE） | ✅ 完成 | toy-synth | GRPO 惰性（SFT 已 peak→组内方差=0）；REINFORCE-w-baseline 恢复信号 |
| **第五轮** | **行为诊断 + 结果审计** | ✅ 完成 | toy-synth | **见下：3 类诚信问题 + agent 输给 RAG + 真实数据崩溃** |
| 第六轮 | 修复底座（真实尺度） | ⏳ 进行中 | synth-v3（待建） | 让读取预算真正有成本 |
| 第七轮 | 真实底座基线 | ⏸ 待启动 | synth-v3 / MuSiQue | RAG 扫描、prompted、oracle path |
| 第八轮 | 真实底座 SFT | ⏸ 待启动 | synth-v3 | 重训（toy 的不迁移） |
| 第九轮 | 真实底座 RL + 诊断 | ⏸ 待启动 | synth-v3 | ratio/add 前沿 + 失败分类 |
| 第十轮 | 增加难度 / MuSiQue 泛化 | ⏸ 待启动 | MuSiQue | 真实 benchmark 落地 |
| 终轮 | 写 AAAI 论文（大白话） | ⏸ 待启动 | — | 汇总 |

---

## 第一轮～第四轮（toy-synth pilot 回顾）

底座为合成语料 `synth_generator.py`：16 文档×10 节，**每节均 ~26 词**，整库 ~4193 词。

- **第一轮**：MuSiQue 检索 Recall@1/5/10 = 0.62/**0.847**/0.90（MRR 0.71）→ 检索可解。Oracle Evidence（给 gold→答）：synth-easy **0.85**（结果A 可解）；MuSiQue（bug 修复前）0.168/0.225。
- **第二轮**：synth-v2 中难度。oracle **0.748**、oneshot_rag **0.609**（改写 0.471）、prompted **0.110**（base 不会用工具）。prompted 失败分类：stop_fail 62.5% / selection_fail 17.5%。
- **第三轮**：SFT v1 学会 search+read(gold) 但**退化性永不 ANSWER**（100% stop_fail，0% 回答）——交叉熵把单条 answer 轮淹没在多条 read/search 轮里。**修复**：answer 轮 4× 上采样 → 回答率 0%→48.5%，group-reward std 0.018→0.316（恢复 GRPO 优势信号）。这是一个干净的「agent 训练病态 + 修复」发现。
- **第四轮**：RL。GRPO 在已 peak 的 SFT 上**惰性**（40 步里仅 4 步有非零组内优势）→ 信号≈0。**修复**：改 REINFORCE-with-baseline（跨实例 EMA baseline）→ 240/240 步非零（信号 ×60）。得到「RL 微弱优于 SFT」的点。

> 以上四轮的价值是**建立了可解性、基线、SFT/RL 管线，并发现两个真实工程病态（SFT 淹没 answer、GRPO 在 peaked 策略上惰性）**。但所有结论都建立在 toy-synth 底座上，第五轮审计证明该底座不足以支撑论文主张。

---

## 第五轮：行为诊断 + 结果审计（本轮，关键）

> 目标：按方法论「先诊断再训练」「失败归因」「结果6：训练好测试崩」全面审计 toy-synth 上的"正向"结果。
> 工具：直接读代码（`docstore.py`/`search_env.py`/`reward.py`）、ckpt adapter_config、n=150 评测 JSON 与 eval log。

### 发现 1：「token」实为词数（度量不诚实）
`docstore.py:48` `token_len = len(self.text.split())`；`search_env.py:147` snippet 成本也用 `.split()`。所有 `read_tokens`/`committed_read_tokens` 实际是**词数**，而非 BPE token。实测 synth 节 BPE/词=1.21。论文核心度量「accuracy-per-read-**token**」名不副实。**必须**：要么统一改为真实 BPE token，要么诚实改称「read words」并标注 token≈1.21×。

### 发现 2：底座玩具尺度 → RAG 饱和、读取预算无杠杆
synth 节均 **26 词**（14–55），整库 4193 词（~5K BPE）。RAG top-5 = 5×40 词 snippet ≈ 把整篇相关内容塞入上下文 → 必然饱和。三条迭代日志（`gamma_sweep`/`maxsteps_sweep`/`iteration_top3`）一致显示：**学到的策略坍缩为「读 1 个候选就答」，对奖励成本旋钮 γ 完全不敏感**（γ=1 vs 10 行为不变；max_steps=3/5/8 行为不变）。原因：读一个 26 词的节几乎免费且 60% 够用 → efficiency-ratio 奖励（论文核心新意）**没有杠杆**。

### 发现 3：模型被错标
`results/frontier_positive.json` 声称 "Qwen2.5-7B-Instruct + LoRA r=32"。实测 `rl-reinforce-{ratio-s60,ratio-s120,add-s120}` 的 adapter base 均为 `ckpts/docscout-sft`，而该 ckpt 的 config 是 hidden=2048/layers=28 = **Qwen3-1.7B**；n=150 评测 JSON 与 eval log 均显示 `model=ckpts/docscout-sft`。**结论：headline 结果实为 Qwen3-1.7B+LoRA，不是 7B。**（对 proposal 的「≤2.5B」反而是好消息，但 frontier_positive.json 字段必须更正。）

### 发现 4（最重要）：仔细看，"正向"结果其实是负面的
oracle_n150 / rag_n150 的 log 确认 reader 用的是 `ckpts/docscout-sft`（1.7B SFT），与 agent 同模型 → **不是混模型对比**。但数值意味着：
- oracle=0.94、RAG top-5=**0.96**；而 RL agent=**0.727**、SFT agent=**0.673**。
- 即 **agent 大幅输给 one-shot RAG**（0.73 vs 0.96）。agent 只读 ~40 词（1 节），RAG 塞 200 词 → 玩具底座上「读得少」直接等于「丢准确率」。
- "RL > SFT" 仅 +0.047~0.067，且 **McNemar p~0.21 不显著**（单种子 n=150）。
- 含义：toy 底座上**不存在成本-质量权衡**（读越多越准，单调），故 RAG（读最多）必胜；read-budget 论点在此**不可测**。

### 发现 5：真实数据上 agent 崩溃（结果6：训练好测试崩）
`hf_agent_sft12_musique_fixed`：synth 训练的 SFT agent 在 MuSiQue（真实、101 词段落、多跳）上 **acc=0.025**（读了 5.83 节、answer_rate 0.95，却几乎全错）。**synth 上学的「读 1 个属性 QA」策略完全不迁移到真实多跳文本。** 目前**没有任何可用的真实数据 agent 结果**。

### 审计结论（第五轮交付）
1. 现有「正向」结果建立在 toy-synth 上，仔细审计为**负面**（agent 输 RAG；RL>SFT 微弱不显著）。
2. 根因单一：**底座太小**使读取预算无意义 → 三类问题（饱和、不显著、不迁移）同源。
3. 方法论奏效：诊断先于训练抓出了 pilot 底座不足，避免了在错误底座上继续堆 RL。
4. **必须先造真实尺度底座并在此上重训**，才能检验 read-budget 论点。→ 进入第六轮。

### 第五轮门控结果（独立子 agent 判定）：✅ AUDIT-PASS
子 agent 逐条核查 5 条断言，均属实。修正：BPE/词比值实测 **1.14**（非 1.21，方向/量级对）。附加发现：
- 模型错标（7B↔1.7B）是**系统性**的，非笔误——投稿若不更正会被复现直接证伪（诚信级）。
- `gamma_sweep` 两端均用 **SFT6 greedy** 当代理（数值一字不差），故"γ 不敏感"其实未真正测到 → **论文不可当作 reward ablation 引用**。
- MuSiQue agent committed_read=125（远高于 synth 的 37-40）却更错 → 失效是「读不懂/选不对」而非「读不够」，佐证泛化崩溃。
判定第五轮完成，进入第六轮。

---

## 子 agent 门控协议（用户要求）

每轮末，派遣一个独立 **general-purpose 子 agent**（无我本轮上下文偏见），给定该轮的「完成判据清单」，让它读相关产物判定 **PASS / FAIL + 理由**。FAIL 则必须补做后再判，不进入下一轮。判据写在每轮正文末。

---

## 第六轮（完成）：修复底座——真实尺度 + 诚实 token + 去饱和验证

**做了什么**：新建 `docscout/data/synth_v3_generator.py`（不破坏旧管线），产出 synth-v3：24 文档×10 节，**节均 ~129 词**（toy 的 5 倍），整库 30897 词（toy 的 7 倍）。关键设计：gold 数值嵌在 intro 段之后（不在前 40 词 snippet 里），节内含 confounder（如"标准层90天/审计365天/缓存7天"）。新增 `scripts/v3_diagnostic.py` 同时报告**词数与真实 BPE token**。

**关键验证（synth-v3 单跳 n=120）**：gold 节在 top-5 检索 100%；gold 数值在 snippet 中 **0%**（RAG 无法扫读）；在完整 gold 节中 100%（读节可答）。

**诊断结果（reader=ckpts/docscout-sft, n=150）**：
| 模式 | acc | ctx 词 | ctx BPE |
|---|---|---|---|
| oracle（gold 全文） | **0.827** | 190 | 239 |
| rag_full k=1 | 0.753 | 146 | 182 |
| rag_full k=3 | 0.813 | 437 | 546 |
| rag_full k=5 | 0.820 | 726 | 907 |
| rag_snip k=5（项目一贯的 RAG） | **0.180** | 220 | 279 |
| rag_snip k=10 | 0.173 | 440 | 556 |
| action_smoke（SFT agent） | answer_rate=1.0, **mean_n_read=3.42（过读）** | | |

**解决了什么**：
1. 底座去饱和——oracle 0.827（可解但非 0.96 饱和）；snippet-RAG 跌至 0.18（去饱和机制生效）。
2. 读取预算**有杠杆**：成本-质量权衡陡峭（rag_full k1→k5：182→907 BPE 仅换 +0.067 acc）；SFT agent 过读 3.42 节 → RL 有明确改进空间（少读、早停）。
3. 诚实 BPE token 已计量（论文核心度量名副其实）。
4. action schema 可迁移（100% 回答，解析正常）。

**还有什么要探究（决定下一轮）**：
- accuracy headroom 很小（oracle−rag_full5=0.007）→ **论文主打必须是"效率前沿"**：自适应读 ~1.5–2 节的 agent 可在 ~270–360 BPE 达 ~0.80，支配 rag_full k3(546BPE)/k5(907BPE)。accuracy 维度 agent 难超 RAG，但 cost 维度有 2–3× 优势。
- 是否需进一步加难（更多文档/更高多跳）以拉大 accuracy 信号？→ 留待第七轮看完真实 agent 前沿点后再定（数据驱动，不一上来就调）。
- RAG 基线定义：rag_full（公平强基线，全文 top-k）vs rag_snip（项目一贯的 snippet）。两者都要报，前沿以 rag_full 为对手。

**第六轮完成判据核查**：①节均129词≥120、文档24≥24 ✅；②BPE token 已报告 ✅；③oracle0.827∈(0.45,0.92) 且 RAG(snip5=0.18, full5=0.82)<oracle ✅；④action smoke answer_rate1.0 ✅。

### 第六轮门控结果（独立子 agent）：✅ ROUND6-PASS
4 条判据全 TRUE。附注：节均词数全语料口径 128.7（中位 131，设计目标 150–400）达标；字面判据脚本只统计每实例首篇文档得 115.7 擦边，不阻塞。确认**成本-质量权衡存在**（rag_full k1→k3 边际 1.6e-4 acc/token，k3→k5 仅 1.9e-5，递减 8.5×）。弱点：oracle(0.827)≈rag_full5(0.82)，RL 的 accuracy headroom 主要在 k1/snippet 端 → 论文主打**效率前沿**。判定第六轮完成，进入第七轮。

---

## 第七轮（进行中）：synth-v3 上训练 SFT agent（CodeScout 式 RFT 暖身）+ 测前沿点

**做了什么**：生成 v3 训练集（seed21，1000 实例，与 eval seed77 值放置独立、无答案泄漏）→ 规则生成 1000 条示范轨迹（search→read(gold)→answer，均 1.15 读/示范，单跳读1/多跳读2，含 4× answer 上采样修复"不答"病态）→ LoRA(r=32) SFT Qwen3-1.7B。

**toy-SFT（toy 26 词节训练）在 v3 上的前沿点（n=150）**：acc=**0.253**、committed 191 BPE、n_read 3.20、answer_rate 1.0。
- **读得不少（3.2 节/191 BPE）却几乎全错** → 在 26 词节上学的策略无法处理 129 词+confounder 的真实节（"读不懂"，呼应第五轮 MuSiQue 崩溃 0.025）。≈rag_snip(0.18) 水平，远低于 rag_full(0.82)。
- **强证：必须在真实底座上重训**（toy 策略不迁移）。

**v3-SFT（在 v3 上重训）**：LoRA SFT，loss 快速→~0（动作格式高度可预测，符合 CodeScout RFT loss→0 再靠 RL 涨的现象）。待评测其 v3 前沿点（预期：学会 read gold→answer，准确率应远高于 toy-SFT 0.253，接近 rag_full 水平但读更少）。

**待探究**：v3-SFT 的准确率/读取量落点 → 是否已能支配 RAG（读更少、准确率相近）？若 v3-SFT 过读，RL（第八轮）应压缩读取；若欠读，RL 应增读。前沿脚本 `scripts/frontier.py` 已就绪，汇总 oracle/RAG/agent 各点 + Pareto。

### v3-SFT 评测结果（n=150，synth-v3）
| 点 | acc | committed BPE | n_read（读动作） |
|---|---|---|---|
| v3-SFT greedy | **0.653** | 165 | 3.25 |
| v3-SFT temp0.3 | 0.647 | 159 | 3.09 |
| 对照 toy-SFT | 0.253 | 191 | 3.20 |
| 对照 rag_full k1 | 0.753 | 182 | — |
| 对照 rag_full k3 | 0.813 | 546 | — |
| 对照 oracle | 0.827 | 239 | — |

**解决/发现**：
1. v3-SFT 从 toy 崩溃 0.253 **恢复到 0.65** → 在真实底座上重训有效，证明 toy→real 失效是训练分布问题而非方法问题。
2. 但 v3-SFT 仍**被 rag_full_k1 支配**（0.653@165BPE vs 0.753@182BPE）：agent 多轮回答比单 shot RAG 低 ~0.10，且 **n_read=3.25（过读）**——只 unique 读到 ~1 个 gold 节（165 BPE≈1.1 节），却发了 3.25 次读动作（多重复读/已读）。
3. **结论**：SFT 给出"能跑但次优"的策略；准确率 + 读取效率都有明显空间 → **RL 有双重杠杆**：提准确率（追 RAG-k1 0.753 / oracle 0.827）+ 砍过读动作（3.25→理想 ~1.5，省延迟/步数）。

**第七轮完成判据**：①v3-SFT 已训并存盘 ✅；②在 v3 上评测得前沿点(acc,bpe) ✅；③确认 toy→real 失效 + 重训恢复 ✅；④揭示 RL 的双重杠杆（提准+砍读）✅。

### 第七轮门控结果（独立子 agent）：✅ ROUND7-PASS（4/4 TRUE）
确认 toy→v3-SFT 恢复（0.253→0.653）；RL 动机成立（acc 差 RAG-k1 10pt + 过读 n_read 3.25）。**科学判断**：v3-SFT 仅被 rag_full_k1 **微弱支配**（acc/BPE 4.06e-3 vs 4.14e-3 近持平），有清晰提升空间（提准 10pt 或砍读到 1）→ read-budget RL 的准确率+效率双动机站得住。判定第七轮完成，进入第八轮。

---

## 第八轮（进行中）：synth-v3 上 RL（reward 消融 ratio vs add）

**做了什么**：从 v3-SFT LoRA 出发，REINFORCE-w-baseline 并行训两个 reward 变体（C2 消融）：
- RL-ratio（GPU1）：`reward=ratio`，efficiency-ratio 成本项（本工作主张）
- RL-add（GPU0）：`reward=additive`，−λ·read_tokens 加性成本（对照）
各 100 步、600 训练实例、temp=1.0、lr=1e-6。`scripts/run_rl_v3.py`。

**信号确认**：RL-ratio var_groups 稳增（每步都有梯度，REINFORCE-w-baseline 克服了 GRPO 在 peaked SFT 上的惰性），reward 范围 -0.2~1.8（健康探索）。

**假设**：RL 相比 v3-SFT 应 (a) 提准确率（朝 RAG-k1 0.753 / oracle 0.827）；(b) 砍过读动作（n_read 3.25→更低）；(c) ratio 比 add 在 iso-accuracy 下读更少（C2）。待两 ckpt 训完评测。

**待探究**：RL 前沿点是否支配 RAG-k1？ratio vs add 谁优？若 RL 提准不显著，考虑加难（更多文档/更高多跳）或多粒度 F1 reward（CodeScout 借鉴）。

### 第八轮结果：RL 在 synth-v3 上未改善前沿（干净负面结果）
| 点 | acc | committed BPE | n_read |
|---|---|---|---|
| v3-SFT | 0.653 | 165 | 3.25 |
| RL-ratio (step60) | 0.653 | 159 | 3.15 |
| RL-ratio (step100) | **0.653** | 161 | 3.21 |
| RL-add (step100) | **0.653** | 163 | 3.20 |

**RL-ratio ≈ RL-add ≈ v3-SFT**（acc/读量逐点几乎不变；分层显示单跳/多跳上 RL 与 SFT **逐实例完全相同**）。reward 消融 ratio=add，**两者都惰性**。REINFORCE-w-baseline 信号健康（var_groups 每步涨，reward 1.3-1.8），但**策略不移动** → SFT 已是 reward-最优（示范"读gold→答"已达 max reward 1.8），RL 无杠杆。

### 第九轮：失败归因（为什么 RL 不涨、agent 输 RAG）
`scripts/v3_failure.py` 在 v3-SFT 上分类失败（n=150）：
- evidence_recall=**0.817**（agent 82% 读到 gold）
- **selection_fail=31% > read_fail=19%** → **主瓶颈是选择（没读到 gold），不是读取/答题**
- 读到 gold 时 acc=**0.718**（接近 oracle 0.827；剩余 0.11 是 confounder 精读难度，答案生成不差）

**核心洞察（论文级）**：read-budget reward（efficiency-ratio）优化的是"读多少/读多省"，但本底座的准确率天花板由**选择**决定（31% 没读到 gold），而选择是**检索/候选**问题，read-budget reward **管不到**。故：
- 单跳题（1 gold 节即够）：无"读多少"决策 → read-budget RL 无杠杆（与 SFT 持平）。
- read-budget RL 只在"需读多节且要决定何时停"的任务上才有用武之地（多跳/合成）。

**待探究（决定第十轮）**：①按单跳/多跳分层，看 RL 是否在多跳（需读多节）上有信号；②若要正面结果，需修复选择瓶颈（rerank / 自适应读候选直到命中 gold）或用多跳主导的更硬底座。

### 第十轮：adaptive 示范 + action_cost reward 正面尝试
1. **adaptive 示范**（按 rank 顺序读候选直到覆盖 gold）：训 SFT-adaptive，评测 acc=**0.573**（**低于** SFT-direct 0.653），选择失败仍 ~33% → adaptive 读取**未修好选择**（agent 学到"读1-2就答"的众数，gold 深时仍漏；多读反而引入 confounder 干扰）。假设负面。
2. **action_cost reward**（每读动作 −0.05，罚浪费的重复读）：已加到 reward，正在 RL-ratio+ac 训练中（验证 action_cost 正确扣减 reward）。待评测是否砍 n_read。
   - 结果：**RL+action_cost acc=0.647, bpe=162, n_read=3.18**（n_read 仅 −0.03，可忽略）→ action_cost **也未能改变贪婪策略**。

### 第十轮总结：四个 RL 变体全部惰性（结论稳健）
| 变体 | acc | bpe | n_read |
|---|---|---|---|
| SFT（基线） | 0.653 | 165 | 3.25 |
| RL-ratio | 0.653 | 161 | 3.21 |
| RL-add | 0.653 | 163 | 3.20 |
| RL+action_cost | 0.647 | 162 | 3.18 |
| SFT-adaptive | 0.573 | 157 | 4.00 |

**所有 read-budget reward 变体（ratio/add/+action_cost）+ adaptive 示范均未能移动贪婪策略**（f3_inertia: 0/150 实例改变）。根因：SFT 策略在 temp=1.0 采样下仍高度 peaked，REINFORCE-w-baseline 的梯度过弱，无法逃离 SFT 局部最优；且奖励在 SFT 最优点附近平坦。**负面结论稳健，停止继续堆 RL 变体（方法论：不为证明有效而无限调）。**

**最终研究结论**：在 1.7B + 真实尺度 synth-v3 上，read-budget RL **未能**让 agent 支配固定 RAG；agent 被 rag_full_k1 支配（选择瓶颈 + RL 惰性）。论文如实报告此负面结果 + 失效机制 + 生效前提。进入论文写作。

---

## 第十一轮（🎉 突破：诊断出惰性根因并修复，RL 终于有效，翻转结论为正面）

> 回应用户关键质疑："是不是环境/奖励没设计好？"——**是的**。深挖出 RL 惰性的两个可修根因，修后 RL 真正改善了前沿。这把论文从"负面诊断"翻转为"正面 + 诊断"。

### 惰性根因（两个，都可修）
1. **奖励信号太弱**：v1 的 efficiency-ratio 在 SFT 最优点平坦（SFT 读 ~1 gold 节已使 eff_ratio≈1），且 action_cost=0.05×3 读=0.15 远小于 answer=1.0 → reward 对读取行为几乎不敏感。
2. **LR 太小**：lr=1e-6 的 LoRA 微更新改不动贪婪解码的 argmax（即使 reward 有微小梯度）。

### 修复
- **非作弊/ grounded 讨论**：topN 示范（读 top-N 不论是否 gold）会训练幻觉（acc 崩到 0.27）——废弃。保留 grounded 的 direct 示范（0.65）作 RL 起点。
- **强奖励**：`action_cost=0.3`（重罚过读，制造 reward 方差）+ `w_gold_hit=0.3` + `w_first_gold=0.5`（重奖读到 gold / 首读命中）+ `w_evidence=0.4`。
- **提高 LR**：lr=1e-6 → **1e-5**（10×），让 LoRA 更新足以改变贪婪策略。
- 从 direct SFT 出发 REINFORCE-w-baseline。

### 结果（synth-v3, n=150, greedy）—— **RL 首次 Pareto 支配 SFT**
| 方法 | acc | committed BPE | n_read | acc/BPE |
|---|---|---|---|---|
| SFT (direct) | 0.653 | 165 | 3.25 | 3.96e-3 |
| **RL strong (s20)** | **0.707** | **128** | **2.81** | **5.52e-3** |
| **RL strong (s40)** | **0.800** | **157** | **2.99** | **5.10e-3** |
| RAG-full k=1 | 0.753 | 182 | — | 4.14e-3 |
| oracle | 0.827 | 239 | — | — |

- **s40 (0.800@157BPE) 同时更准(>RAG-k1 0.753)且更省(<182BPE) → Pareto 支配 one-shot RAG**。✅✅ **论点最强结果成立**。
- 轨迹 SFT 0.653 → s20 0.707 → s40 0.800：RL 随步数单调改善准确率，成本维持在 RAG-k1 之下。
- acc/BPE 5.1–5.5e-3 > RAG-k1 4.14e-3 → accuracy-per-read-token 效率反超 RAG。
- 训练 reward 有真实方差（0.5–2.27，含负值），var_groups 每步涨 → 梯度有效（vs v1 的惰性）。

### 最终收敛结果（s100/s120，n=150，样本人工核实为真实正确）
| ckpt | acc | BPE | n_read |
|---|---|---|---|
| RL strong s100 | **0.840** | 170 | **1.00** |
| RL strong s120 | **0.833** | 170 | **1.00** |

- **n_read=1.00**：agent 学会**搜→只读 1 个 gold 节→精确答**（理想 read-budget 策略）。样本核实：读 1 节精确抽出 "product owner / 10 / 365 / 8 / QuartzLedger"（非评分 artifact）。
- **s100 (0.840@170) 支配 RAG-k1 (0.753@182)**：更准且更省；甚至超 oracle(0.827)——RL 优化了抽取（oracle 用 SFT-reader 较弱）。
- **选择瓶颈被修复**：v1 的 31% selection_fail → strong-s100 n_read=1 说明 agent 学会直接读对 gold 候选（failure 分类量化见下）。
- 确认种子(seed-2) s20 评测中；轨迹 s20→s120 全程 > SFT，方向稳健。

### 第十一轮门控结果（独立子 agent）：✅ ROUND11-PASS
正面结果真实成立：RL-strong seed1 s100/s120 = 0.840/0.833@170BPE/n_read1.0；seed2 final = 0.82@170（2 seed 一致）。支配 SFT(0.653@165) 与 RAG-k1(0.753@182)。机制真实（非评分 artifact）：ev_recall=0.843(>RAG 75%)、acc|read_gold=1.0、read_fail=0；抽查 pred/gold 真实匹配。论文 8 页正面框架编译无误。**"诊断→修复→验证"链条成立，目标（反复迭代直到 RL 有效）达成。**

---

## 第十二轮（进行中）：建多跳链式评测 synth-v4，拉开 RAG-agent 差距

> 用户洞察：synth-v3 的 RAG 能到 0.75-0.82，说明评测太简单；应设计需跨多文档链式推理的问题（如 5 跳），让 RAG 难以达成，从而突出 agent 多步读取的优势。

### 底座设计（`docscout/data/synth_v4_generator.py`）
**K-跳链式问题**：例 "rate_limit 是多少？该系统的 backup_freq 等于另一系统的 retention，后者 sync_delay 等于…直到 DeltaSync 的 retry_limit 等于其 retention"。每跳的答案决定下一跳的检索目标。
- 30 文档 × 6 数值属性，共享同一组 30 个唯一值（每属性按不同排列分配）→ 链式链接"属性 L=值 v"总有唯一解。
- 问题用嵌套描述，**只命名起始实体+属性，不透露中间值** → RAG 一次性检索无法跟随链条。
- hop 分布：2跳30% / 3跳40% / 4跳20% / 5跳10%。gold = 链上 K+1 个 section。

### 诊断结果（v4, n=150, reader=1.7B-SFT）—— **RAG 崩溃，目标达成**
| 模式 | acc |
|---|---|
| oracle（给全 gold 链） | **~0.65**（可解但 1.7B 多跳推理有难度，未饱和） |
| rag_full k=1 | ~0.03 |
| rag_full k=3 | **0.020** |
| rag_full k=5 | **0.027** |
| rag_snip | ~0.02 |

- **RAG 即使 k=5 全文也仅 ~0.02-0.03**（一次性检索无法发现中间值、跟随链条），**oracle 0.65** → headroom 巨大（+0.63）。
- 这正是用户要的"更难评测"：RAG 远低于 0.8，agent 若能多步跟随链可大幅领先。
- 链可解性已验证（实例追踪：DeltaSync retention=677 → Z.retry_limit=677 → Z.backup_freq=677 → Y.sync_delay=677 → Y.retention=872 → X.backup_freq=872 → X.rate_limit=856 ✓）。

**待做**：构建链式跟随 SFT 示范（search 链值→read 跨文档→合成答案）→ 训练 → 预期 agent 远超 RAG(0.02)，朝 oracle(0.65)。

### 结论修正
**read-budget RL 不是"本质不行"——是 v1 奖励太弱 + LR 太小导致"假装在学实则不动"。** 修好后 RL 同时提准+省读，支配 SFT 并在效率上反超 RAG-k1。论文从"诚实负面诊断"升级为"**正面结果 + 完整诊断（为何 v1 失败、如何修、修后 work）**"，更强。继续追踪 s40/s100 前沿、多 seed、写论文。

### 全局前沿（synth-v3, n=150，`scripts/frontier.py`）
| 方法 | acc | read BPE | acc/kBPE | Pareto? |
|---|---|---|---|---|
| agent 簇（SFT/RL-ratio/RL-add） | ~0.65 | ~160 | 4.0 | ❌ 被 RAG-k1 支配 |
| SFT-adaptive | 0.573 | 157 | 3.65 | 边缘 |
| **rag_full_k1** | **0.753** | 182 | 4.13 | ✅ |
| oracle | 0.827 | 239 | 3.46 | ✅（上限） |
| rag_full_k3 | 0.813 | 546 | 1.49 | |
| rag_full_k5 | 0.820 | 907 | 0.90 | |

**核心结论（论文级，诚实负面）**：在 synth-v3 上 read-budget agent **被 one-shot RAG(k1) 支配**。根因机制：
1. **agent 的选择比 BM25 top-1 还差**：agent 31% 没读到 gold（学到"读1-2就答"），而 rag_full_k1 直接读检索 top-1（含 gold ~75%）→ agent 准确率上限被选择卡死。
2. **read-budget reward 管不到选择**：efficiency-ratio 只罚 committed token（agent 已读~1节，很省），不罚浪费的读动作、也不教"读哪个候选"→ SFT 已 reward-最优 → RL 惰性。
3. **read-budget RL 的真实生效前提**（论文 take-away）：需 (a) 选择不是瓶颈（检索强或 agent 选得准）、(b) 策略有"读多少"的真实自由（任务需读多节且解不唯一）。synth-v3 单跳主导，两者皆不满足。

**待探究**：action_cost RL 是否砍 n_read（小效率正面）；是否在多跳主导或检索更强的设置下 agent 能反超 RAG（未来工作）。
