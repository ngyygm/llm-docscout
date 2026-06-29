# 数据环境设计（升级版）— 真实语料为底 + 反向构造 + 严格校验

> 回应用户关键追问：以开源数据集为**底座**，在其上"反向构造"保证必用某些段落的任务，并做必要性/充分性校验。直接消灭评审 W4（纯合成=玩具数据）。

## 0. 总策略
两条腿：
1. **Grounded reverse-construction（主，出真实结果）**：真实文档语料为底 → 抽样证据段落 → 反向生成 query/任务 → necessity/sufficiency 过滤。
2. **Synth（保留，快速调试/隔离消融）**：已实现的 `docscout/data/synth_generator.py`，可控、秒级生成、无外部依赖，用于 env/reward 调试与小消融。

## 1. 候选底座数据集（已查证）

| 数据集 | arXiv | 规模 | gold 证据粒度 | 适配度 |
|---|---|---|---|---|
| **MuSiQue** | 2108.00573 | 25K | 支撑**段落**，2–4 跳，组合时强 filter | ⭐⭐⭐ 必要性靠构造保证 |
| **HotpotQA** | 1809.09600 | 113K | **句级** supporting facts | ⭐⭐ 规模大、经典 |
| **DocScope** | 2605.08888 | 长文档 | 层级证据链 page→region→fact→answer | ⭐⭐⭐ 长文+可验证，最贴 section 导航 |
| **2WikiMultiHopQA** | 2011.01060 | — | 推理路径三元组 | ⭐⭐ 结构化 |

**推荐主底座：MuSiQue**（必要性靠构造保证，证据=段落=我们的 section）；**长文档轴加 DocScope**；规模不够时 HotpotQA 补量。

## 2. 反向构造管线（核心）
```
base 语料 (Wikipedia 段落 / DocScope 长文 / 用户自有文档)
   │
   ▼  ① 按 (doc, section/paragraph) 建索引
   ▼  ② 抽样证据集合 E = {s_1..s_k}（控 k=跳数；混入 distractor 段落 D）
   ▼  ③ LLM 反向生成 query/任务指令 q，使答案 a 只能由 E 推出
   ▼  ④ necessity/sufficiency 校验（见 §3）→ 不过则丢弃/重生
   ▼  ⑤ 输出 QAInstance{question=q, gold_answer=a, gold_evidence=E(section-locatable), corpus=E∪D}
```
- 关键：**distractor D 必须与证据同域**（同主题/同文档邻居段），否则 search 太容易，pilot 信号失真。
- 多跳：k≥2 时 q 需跨段推理（MuSiQue 式）；单跳 k=1。
- 任务指令可不止 QA：可生成"定位/抽取/比较"多种（呼应 CodeScout 的"定位"任务多样性）。

## 3. 校验协议（保证数据对）
| 校验 | 做法 | 不通过的处置 |
|---|---|---|
| **必要性 necessity** | (a) 闭卷（不给文档）能否答对；(b) 只给 distractor（去掉 gold E）能否答对 | 任一答对→歧义，丢弃/重生 |
| **充分性 sufficiency** | 只给 gold E 作上下文，reader 模型能否恢复 a | 不能→证据不自足，重生 |
| **答案确定性** | gold a 必须 EM 可判（数值/实体）；自由文本用归一化+LLM-judge | 多义→重生或降级 |
| **section 可定位** | gold 必须映射到具体 section_id | 构造时保证 |
| **零污染** | train/eval 的**底层文档**不重叠 | 切分前先分语料 |
| **统计** | #证据段/跳数/文档长度/答案类型 分布 | 报告 + 监控偏斜 |

> MuSiQue 的 answerability filter 就是 necessity/sufficiency 的工业实现，我们直接复刻其思想。

## 4. 规模与划分（参考 CodeScout）
- CodeScout：**39K 训练 / 128 仓库**，模型 Qwen3-1.7B/4B/14B，eval SWE-Bench Verified/Pro/Lite，**仓库零重叠**。
- s3：只用 **2.4K** 就训出搜索 agent（下限）。
- **DocScout（≤2.5B，Qwen3-1.7B/2.5B）建议**：
  - **先切底层语料**：train-docs ≈90% / eval-docs ≈10%（零文档重叠，对齐 CodeScout "no repo overlap"）。
  - **Pilot（R0–R2）**：≈ **2–5K** 训练实例（快速验 reward 上升、前沿方向）。
  - **Full**：≈ **10–20K** 训练（CodeScout 39K 为上限参照）。
  - **Dev**：≈1–2K（eval-docs，前沿测量/奖励 shaping）。
  - **Test（外部诚实评测）**：≈1–2K（eval-docs）**+ 原始 MuSiQue/HotpotQA/DocScope 官方 test**（泛化报告，避免自评）。

## 5. 与现有 synth generator 的关系
- 保留 `synth_generator.py`：env/reward 冒烟、消融隔离（可控变量）、CI 单测。
- 新增 `grounded_generator.py`（待实现）：跑 §2 管线，产出真实结果用的训练/评测集。
- 两者**共享 `QAInstance` schema + DocStore**，env/reward/rollout 完全复用。

## 6. 待用户拍板的叉路（见下问）
1. 主底座选哪个（MuSiQue / HotpotQA / DocScope / 用户自有政策·手册文档）？
2. 现在就实现 grounded 管线，还是先把 env/reward/rollout 在 synth 上跑通（R0）再换真实数据？
