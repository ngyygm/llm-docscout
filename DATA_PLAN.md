# DATA_PLAN.md — DocScout 数据集与环境整理计划

> 状态基线日期:2026-06-29。本文件把外部 review 的建议**对齐到仓库真实代码状态**,
> 标注每条「✅已做 / 🟡部分做 / ❌未做」,避免重复造轮子;并按「先解锁论文结论、
> 再扩面」的顺序排优先级。配套结论审计来源:`refine-logs/round-result.md`。

---

## 0. 两个硬约束(影响所有方案)

1. **算力**:单卡 **NVIDIA H20D(143GB)**,远程 `ssh root@dev-H200x1`,**无外网**。
   - policy 模型 ≤2.5B,143GB 显存非常宽裕 → corpus / 上下文可以放大(见 §4 scale)。
   - 任何联网下载(HF 数据集、模型、reranker 权重)必须**本机下好再 scp 过去**。
2. **GLM-5.2 API(host 端,最长 400k 上下文)**:用于**合成数据生成、LLM-as-judge、
   closed-book audit**。**不是 policy 模型**。400k 上下文 → 可一次性把整个 corpus
   喂给 GLM 做「整库生成 / 全局一致性校验 / oracle 解题」,无需分块。
   - 封装见 `docscout/llm/`(`load_config()` + `LLMClient`),key 不入 git。

---

## 1. 现状核对:review 假设 vs 仓库真相

外部 review 基于的若干前提**已过时**。核对结果:

| review 建议 | 仓库真实状态 | 结论 |
|---|---|---|
| §3 共享 corpus + `corpus_id` 引用 | `docscout/data/corpus_store.py`:`CorpusStore`/`QAInstanceRef`/`serialize_separate` | ✅ **已实现**,但生成器尚未默认走它(见 §3) |
| §3.3/§11 防泄露、按实体/属性切分 | `corpus_store.validate_split_leakage()`(实体/doc/section 三级) | 🟡 函数已实现,**未接入数据生成流水** |
| §13 版本冲突反作弊集 | `synth_v5_generator.py`(deprecation marker + 当前/旧双文档) | ✅ **已实现** |
| §16 五个诊断实验 | `scripts/diagnostic_{closed_book,snippet_only,oracle_path,template_holdout}.py` | 🟡 **4/5 已有脚本**,缺 candidate-scaling;且需重跑 |
| §5 answer normalization | `reward/answer_scoring.py`:normalize + 单值数字匹配 | 🟡 有基础,**缺别名/区间/LLM-judge** → 用 GLM 补 |
| §8.1 sentence/span-level evidence | reward 仅 section-F1(`evidence_f1_read`) | ❌ **未做** |
| §9.2 snippet 长度消融 | `DocStore(snippet_words=40)` 可配 | 🟡 参数可调,**未做系统消融** |
| §9.1 search top_k 可变 | `EnvConfig.search_k` 可配 | 🟡 同上 |
| §9.3 read-by-idx | `SearchEnv._do_read` 支持 `idx=` | ✅ **已实现** |
| §9.4 invalid action 惩罚 | env 对 NOT FOUND / ALREADY READ / OUT OF RANGE 返回提示但**不计入 reward** | 🟡 有观测,**reward 未惩罚** |
| §2 v1/v2 旧数据归档 | `data/synth/` 目录**根本不存在**(被 .gitignore + 未生成) | ⚠️ 前提不成立(见 §2) |

**一句话**:团队比 review 以为的走得更远(corpus_store / v5 / 诊断脚本都已有),
但**这些能力大多没接进主流水**,且 `data/synth/` 数据未纳管。

---

## 2. 数据现状与「版本治理」(对应 review §1/§2)

### 2.1 磁盘实况
```
data/grounded/   45MB  MuSiQue train/dev/test/2k                 ✅ 真实多跳
data/sft/       125MB  20 个 SFT 轨迹 jsonl(v2..v12 + synth_rug) ✅ 训练轨迹
data/personamem/ 35MB  PersonaMem(真实长上下文对话 QA)          ✅ review 未提及的真实源
data/synth/      ——    不存在(.gitignore 忽略 + 未生成)        ❌ 需本机重生成
```
> 注:review §2 要「归档 `clean_eval.json`/`v2_dev400.json` 等」——这些文件**不在仓库**,
> 是早期 session 的产物。生成器(`synth_generator.py` v1、`synth_v3/v4/v5`)都在,可复现。

### 2.2 行动:建立 `data/dataset_registry.json`(❌新建)
单一事实源,登记每个数据集的:`id / version / generator+args+seed / split 策略 /
n_instances / 用途(main|smoke|regression|diagnostic) / 状态(active|archived) / 创建命令`。
- 生成器统一加 `--registry` 钩子,产出数据时**自动追加一条 registry 记录**(含 seed)。
- 旧 v1/v2 定位为 `smoke/regression`;v3=`single-hop hard`;v4=`chain main`;v5=`conflict`。

---

## 3. 数据格式:共享 corpus 落地(对应 review §3)

`corpus_store.py` 已实现但**生成器仍在每条 QAInstance 内嵌完整 docs**
(`synth_v4_generator.serialize` 把 `docs` 整份写进每条记录)。

**行动(🟡→✅)**:
1. 生成器输出改为 `serialize_separate()` 的双文件:`corpora.json` + `qa.jsonl`(ref)。
2. 训练/评测入口用 `deserialize_separate()` 物化。
3. **一个 corpus 对应 N 条问题**(review §3.3):v4/v5 生成器当前其实已是「同一 corpus
   多问题」,但内嵌存储掩盖了这点 → 改 ref 存储后天然体现「共享世界 + 背景噪声文档」。
4. 切分前跑 `validate_split_leakage()`,把报告写进 registry。

---

## 4. 底座 scale:**这是解锁论文结论的最高优先级**(对应 review §10)

> `round-result.md` 第五轮审计的核心负面结论:**toy 底座上 read-budget 杠杆不存在**。
> synth 段落均 26 词、整库 4193 词 → RAG top-5 直接把全部相关内容塞进上下文 → 必然饱和;
> γ(1 vs 10)、max_steps(3/5/8)**行为不变**,efficiency-ratio 奖励**无杠杆**;
> agent 0.73 **大幅输给** one-shot RAG 0.96;RL>SFT 仅 +0.05 且 McNemar p≈0.21 不显著。

**没有 scale,换再多关系类型也测不出读取预算优势。** 三档(H20D 显存充裕,可直接上 medium):

| 档位 | section 词数 | corpus 规模 | 整库词数 | 用途 |
|---|---|---|---|---|
| small  | 100–300 | 30 docs × 8 sec | ~5K | debug/smoke(现状) |
| **medium** | 300–800 | 100 docs × 10 sec | ~300K–500K | **主实验** |
| large  | 800–2000 | 500+ docs | 数 M | 压力测试 / RAG 成本爆炸论证 |

**要观测的曲线(才是论文卖点)**:corpus↑ → one-shot RAG 的 read_tokens 与成本**暴涨**,
而 agent 的 committed_read_tokens **保持低位**且准确率不崩 → read-budget 策略价值随规模显现。
- **GLM-5.2(400k 上下文)在此处的作用**:large 档整库可一次喂给 GLM 当 oracle 上限、
  或生成跨文档链;也可作 RAG-with-huge-context 的对照(把「无脑全读」成本量化)。

**配套必修(`round-result.md` 发现1)**:`token_len` 当前是 `.split()` **词数**,
非 BPE token(实测差 1.21×)。论文核心度量「accuracy-per-read-**token**」名不副实。
→ **统一改真实 BPE token**(用 policy tokenizer),或诚实改称「read words」并标注。
建议改 BPE(`docstore.py:47` 的 `token_len` 与 `search_env.py` snippet 成本两处)。

---

## 5. v5 骨架升级:从「数值接龙」到异构企业文档(对应 review §4/§14)

review §4 的批评对 **v4** 成立:6 个数值属性共享同一值集,链靠「数值等值跳转」,
agent 可能学到 `search(数字)` 模板而非通用检索。**保留 v4 作机制验证集,新建 v5+ 骨架。**
(注:现有 `synth_v5_generator.py` 是「版本冲突」集,定位不变;此处指**链式骨架**的下一版,
建议命名 `synth_v6_hetero` 以免与现有 v5 冲突。)

**用 GLM-5.2 生成**(400k 上下文可整库生成、保证全局 ID 桥接一致):
- 链接关系多样化:数值等值(保留少量)+ **字符串别名桥**(routing code `R3`→owner)+
  类别链 + 时间版本链 + 条件链(region=EU/US)。
- 中间桥接值用**业务标识符**(`RET-A7`/`Blue Finch`/`regulated-medium`),非裸数字
  → 杜绝 `search(5)` 捷径;并埋干扰项(changelog 旧值、former owner)。
- 答案类型多样化:number/date/team/code/yes-no/short-phrase/set-valued/**unanswerable**。
- 问题类型配比(review §14):single 20% / single+condition 15% / 2-hop 25% /
  3-hop 15% / version-conflict 10% / exception 5% / **no-answer 10%**。

**关键:每条 GLM 生成的数据必须可程序校验**(桥接 ID 唯一可解、gold 段确含答案、
no-answer 集确无答案)——校验器 `scripts/verify_synth.py`(❌新建),不可解的丢弃。

---

## 6. 真实评估扩面(对应 review §6/§12)

现状只有 MuSiQue + PersonaMem。目标评测矩阵:

| 数据源 | 作用 | 状态 | 获取 |
|---|---|---|---|
| synth v3 | 单跳 read-required | ✅ 生成器在 | 本机生成 |
| synth v4 / v6-hetero | 可控多跳策略 | ✅/❌ | 生成 |
| MuSiQue | 真实多跳 | ✅ 在 | — |
| HotpotQA / 2WikiMultiHop | 多跳泛化 | ❌ | **本机下→scp** |
| Qasper / NarrativeQA / QuALITY | 长文档阅读 | ❌ | **本机下→scp** |
| Natural Questions / TriviaQA | 检索/记忆风险 | ❌ | 本机下→scp |
| 企业文档合成集(v6) | 真实工具场景 | ❌ | GLM 生成 |
| no-answer set | 停止/拒答 | ❌ | GLM 生成 |

**MuSiQue 泄露 closed-book audit(review §12)**:`diagnostic_closed_book.py` 已存在,
需对 MuSiQue 跑三档:(a) 仅问题、(b) 问题+随机错误文档、(c) 实体改名为 synthetic。
预期:v4≈0;MuSiQue 若 closed-book 偏高 → 标注「检索 vs 记忆」风险。**用 GLM 当被测/judge。**

---

## 7. reward 精修(对应 review §8)— env 回归后做

| 项 | 现状 | 改法 | 优先级 |
|---|---|---|---|
| §8.1 evidence 太粗 | 仅 section-F1 | 加 sentence/span-level:gold_evidence 增 `sent_id`/`answer_span`;reward 拆 `section_hit`/`sentence_hit`/`answer_supported` | 高(scale 后) |
| §8.2 efficiency_ratio 不稳 | `committed_gold/committed_read` | 改 `relative_cost = agent_read / oracle_min_read`,仅在「答对+证据对」时给 cost bonus | 高 |
| §8.3 termination 太弱 | submit +0.5 / 耗尽 -0.5 | 四分类:correct+support +1 / premature -2 / exhausted -2 / 充分后续读 -0.5/次 | 中 |
| §8.4 gold_hit 致穷举 | `w_gold_hit` 加分 | 改 `first_gold_rank`:首读 gold +2 / 前2 +1 / 之后 +0.2 | 中 |
| §9.4 invalid action | env 提示但不罚 | reward 对 NOT FOUND/ALREADY READ/OUT OF RANGE/空 query/重复 query 计惩罚 | 中 |

> 注:env 已有 `committed_read_tokens`/`gold_tokens_read(committed_only=)` 等账目,
> 改 reward 不必动 env;但 sentence/span 需要 gold 标注升级(生成器 + MuSiQue 适配)。

---

## 8. 环境消融(对应 review §9)— 证明「不是调参适配」

`EnvConfig` / `DocStore` 参数都可配,**缺的是系统化消融脚本 + 报告**(❌新建 `scripts/ablate_env.py`):
- snippet 长度:20 / 40 / 80 / title-only / title+first-sent。
- search top_k:3 / 5 / 10。
- chunk 切分:短 80–120 / 中 200–400 / 长 800–1200 / natural / sliding-window。
- 检索器:BM25 / BM25+rerank(`RerankRetriever` 已实现,需本机下 ms-marco-MiniLM)/
  dense / hybrid / oracle。**dense/hybrid retriever 需实现 + 本机下 embedding 模型。**
- baseline 矩阵(review §7.2):one-shot RAG / fixed search-read-answer / prompted /
  SFT / SFT+RL / oracle-path / oracle-evidence。**fixed-agent 必做**(若它已够好则 RL 必要性弱)。

---

## 9. 反作弊评测集(对应 review §13)— 用 GLM 生成

独立小集,专测投机(每类 ~100 条,可程序校验):
1. **snippet 泄露**:snippet 像答案,完整 section 否定它(测「是否 read」)。
2. **旧版本干扰**:已由 `synth_v5` 覆盖 🟡(扩量即可)。
3. **同名实体**:AuroraPay EU/US/Sandbox,问 EU 干扰 US。
4. **无答案强相关**:全相关但无答案(测拒答)。
5. **错误 bridge**:current owner vs former owner,问 current。

---

## 10. 执行顺序(诊断先行,one-factor-at-a-time)

遵循 `refine-logs/自动化实验迭代方案.md` 的「先诊断再训练」。**阶段 0 不依赖 GLM,
可立即并行启动;GLM 在阶段 2 起重度使用。**

**阶段 0 — 解除阻塞 & 复现负面审计(本周,最高优先级)**
- [x] 补回 `docscout/env/`(已 merge,7/7 单测过)
- [x] 封装 GLM API(`docscout/llm/`,smoke 全过)
- [x] 本机重生成 v3/v4/v5 → 建 `data/dataset_registry.json`(已登记 7 个数据集)
- [x] 修 `token_len` → 真实 BPE token(`docscout/env/tokenizer.py`;tiktoken 离线 / Qwen 可 pin;
      实测 BPE/word≈1.3–1.5;rollout smoke read_tok 由 ~40词 → ~368 BPE,度量正名)
- [x] **oracle_path 诊断(host 端,无需模型)**:修复脚本 3 个 bug 后跑通 v4_eval300:
      - **真实发现**:77/300 链「oracle 不可达」,全部是 **裸数字 search-miss**
        (如 `search('177')` 未在 top-5 召回目标实体)→ **实证 review §4.3**:数值接龙链
        即使对 oracle 也脆弱/不自然 → 强化「v6 改用业务标识符桥接」的必要性。
      - 93/300 违反 static max_steps(8)、0 违反 dynamic;leak 251 minor/46 mod/3 severe。
- [ ] closed-book / snippet-only / template-holdout:**需 GPU 模型(HFClient)+ SFT ckpt**,
      留待 GPU 任务(本机无法跑)。可选:用 GLM API 改写一版 host 端 closed-book。

**阶段 1 — scale up(解锁论文结论的关键)**
- [ ] medium 档生成器(100 docs × 300–800 词,共享 corpus ref 存储)。
- [ ] 画「corpus 规模 vs RAG 成本 / agent read_tokens / 准确率」曲线 → 证明杠杆出现。
- [ ] candidate-scaling 诊断(补 review §16 实验4)。

**阶段 2 — v6 异构骨架 + 真实扩面(GLM 重度使用)**
- [ ] GLM 生成 `synth_v6_hetero` + `verify_synth.py` 校验。
- [ ] 本机下 HotpotQA/2Wiki/Qasper → scp;接 `grounded_generator` 适配。
- [ ] MuSiQue closed-book audit;LLM-judge 接入 `answer_scoring`(GLM 兜底语义判分)。

**阶段 3 — reward 精修 + 环境消融 + 反作弊集**
- [ ] §7 reward 四项;§8 消融脚本 + baseline 矩阵;§9 反作弊集。
- [ ] 实验有效性矩阵(review §15)写进报告。

---

## 11. 待用户确认的取舍

1. **scale 优先 vs v6 骨架优先**:我建议**先 scale(阶段1)**——因为审计证明当前一切正向
   结论都因 toy 底座不可信;scale 后 read-budget 杠杆才可测。v6 骨架放阶段2。是否同意?
2. **token 度量**:改真实 BPE token(推荐,度量正名)还是改称「read words」标注 1.21×?
3. **真实数据下载清单**:先下哪几个?(建议最小集:HotpotQA + 2Wiki + Qasper)
4. **v6 命名**:现有 `synth_v5` 是版本冲突集;新异构链式骨架我暂命名 `synth_v6_hetero`,可否?
