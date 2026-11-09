# Jev-like 模型可微调性实测

> 2026-11-09｜本地 RTX 4060 8G 实测｜结论：**Jev 官方不可微调，Laya / Kev 可以，且已跑通**

---

## 一、先划清界限：Jev 本身没有开源

查阅 TypeSafe 官方发布页、文档、代码仓库，**没有找到 Jev 模型权重和完整训练配方的下载入口**。
公开的只是适配器、Skill 等外围代码，以及托管的 API 服务。

所以「微调 Jev」这个说法不成立。能微调的是社区复刻的 **Jev-like 模型**。

官方训练方法叫 **RLCD**（Reinforcement Learning for Calibrated Decisions），
即用严格适当评分规则（proper scoring rules）做强化学习，优化决策质量与概率校准——但配方未公开。

---

## 二、可微调的候选（按显存从小到大）

| 项目 | 模型 | 参数量 | 底座 | 训练方式 | 显存 | 许可证 |
|---|---|---|---|---|---|---|
| **Laya** | laya / laya-multilingual | 421M / **322M** | ModernBERT | 全参或冻结编码器，RLCD | **≥12 GB（全参）/ 8 GB（冻结）** | Apache-2.0 |
| **Kev** | kev-0.8b | 0.8B | Qwen3.5-0.8B | 冻结底座 + **rank-16 LoRA** + pointer head | **4 GB** | Apache-2.0 |
| **Kev** | kev-4b | 4B | Qwen3.5-4B | 同上 | 32 GB 级 | Apache-2.0 |
| **Kev** | kev-9b | 9B | Qwen3.5-9B | 同上 | **17 GB** | Apache-2.0 |
| **Kev** | kev-27b | 27B | Qwen3.8-27B | **全权重微调** | 66 GB（80G 卡） | Apache-2.0 |
| **Nimble** | nimble-9b | 9B | Qwen3.5 | **LoRA** | 20 GB 级 | Apache-2.0 |
| **AgentJev** | agent-jev-0.6b | 0.6B | Qwen3-0.6B | 全参（去 LM head + 打分头） | 34 GB（官方） | Apache-2.0 |
| **open-jev-typed** | 150M | 150M | — | — | **Colab T4 30 分钟** | — |
| **SemIf / Simple Jev** | 不训练 | — | 任意 LLM | 只改推理（读候选 logit） | 推理显存 | — |

**关键**：Laya 和 Kev 都提供完整训练代码，不是「只能推理」的封装。

---

## 三、本地实测（RTX 4060 8G）

### 3.1 零样本中文效果（未微调）

模型：`convaiinnovations/laya` 的 `multilingual` 子集（322M，Apache-2.0）
加载 12.6 秒，推理 **22~24 ms/条**（首条 3.5 s 是预热）。

| 公告标题 | 判断 | 概率 | 是否正确 |
|---|---|---|---|
| 关于控股股东、实际控制人股份减持计划的预披露公告 | 利空 | 72.4% | ✅ |
| 预计2024年度净利润同比增长120%至150% | 利好 | 91.0% | ✅ |
| 关于收到深圳证券交易所关注函的公告 | 利空 | 87.7% | ✅ |
| 关于以集中竞价方式回购公司股份的进展公告 | 中性 | 48.5% | ⚠️ 偏保守（回购是温和利好） |
| 关于变更公司注册地址的公告 | 中性 | 57.0% | ✅ |

**5 条里 4 条正确，1 条偏保守。零样本就有这个水平，说明底座的中文金融语感可用。**

输出是结构化的：

```json
{"answers": {"sentiment": {
    "type": "choice", "choice": "利空",
    "probabilities": {"利好": 0.089, "中性": 0.187, "利空": 0.7239},
    "confidence": 0.3057, "answer_confidence": 0.7239}}}
```

注意 `probabilities` 是三项归一化的分布，`answer_confidence` 是选中项的边际概率——
**这正是 Jev 的核心卖点：一次前向给出可校准的分布，而不是生成字符串再解析。**

### 3.2 微调实测

训练命令（CLI 是随 pip 包发布的，`laya.train_cli`）：

```bash
python -m laya.train_cli \
  --data data/laya/demo_train.csv --eval data/laya/demo_eval.csv \
  --out runs/laya_head --question-id sentiment \
  --instructions "判断这条A股公告对股价短期的影响方向" \
  --epochs 10 --micro-batch 4 --grad-accum 2 --freeze-encoder --device cuda
```

数据格式就是 CSV 两列：`text,label`。

**三次尝试的显存结果：**

| 配置 | 结果 |
|---|---|
| `micro_batch=8, grad_accum=8`，全参 | **OOM**（已占 6.66 GB，分配失败） |
| `micro_batch=2, grad_accum=4`，全参 | **OOM**（6.74 GB） |
| `max_len=256`，全参 | **OOM**（6.74 GB，max_len 不影响基础占用） |
| **`--freeze-encoder`（只训分类头）** | **成功**，10 epoch，loss 1.05 → **0.72** |

**结论：322M 全参微调的显存门槛是 12 GB 左右（模型 1.3 + 梯度 1.3 + Adam 2.6 + 激活），8 GB 只能冻结编码器。**

### 3.3 为什么 accuracy 没提升

19 条示例数据、6 条 eval（且与训练集重叠），模型只能记住这 18 条，学不到规律。
**这不是模型的问题，是数据不够。** 真实微调需要 1,000~5,000 条带标签数据。

---

## 四、硬件方案

| 场景 | 配置 | 能跑什么 | 参考成本 |
|---|---|---|---|
| **本地验证** | 4060 8G（现有） | Laya 冻结编码器微调、零样本推理 | 0 |
| **小规模微调** | 4090 24G | Laya 全参、Kev-0.8B/4B LoRA | 约 2~3 元/小时 |
| **对标 Jev** | A800 80G | Kev-9B、Kev-27B 全权重 | 约 8~12 元/小时 |
| 极限 | B200/H200 | Kev-27B 舒适 | 更贵 |

**时间估算（1 万条数据）**：

| 模型 | 4090 | A800 |
|---|---|---|
| Laya 322M 全参 | 5~10 分钟 | 3~5 分钟 |
| Kev-0.8B LoRA | 30~60 分钟 | 15~30 分钟 |
| Kev-4B LoRA | 2~3 小时 | 1 小时 |
| Kev-9B LoRA | 跑不动（17G 勉强） | 2~3 小时 |

**建议：先在 4060 上把 Laya 冻结版跑通（零成本），出结果后再决定要不要租 4090 做全参。**

---

## 五、真正的瓶颈是数据，不是模型

微调的输入是 `(state, question, label)`：

```json
{"text": "关于控股股东股份减持计划的预披露公告", "label": "利空"}
```

三个待解决问题：

1. **数据来源**：公告标题我们有渠道（`stock_notice_report` 被限流，但业绩预告 9.1 万条已在本地；
   公告标题需另找，或用「业绩预告 + 变动幅度」构造）
2. **标签怎么来**：LLM 弱标注（DeepSeek/GPT 批量打标）+ 人工抽检 500 条算 Cohen's kappa
3. **不能自己给自己出题**：用「未来收益」当标签是循环论证，报告里会被判废

**推荐路径**：业绩预告（9.1 万条，标签是交易所强制的预增/预减）→ 直接映射成利好/利空/中性
→ 抽 3,000 条做微调 → 剩下做测试集。**这条路零标注成本。**

---

## 六、环境坑（必须记住）

**`pip install laya` 会把 torch 从 2.7.1+cu118 升级到 2.15.0+cpu，CUDA 直接消失。**

连带破坏：
- `torchvision` 被升到 0.26.0（要求 torch 2.11.0），导入 `transformers` 时报
  `RuntimeError: operator torchvision::nms does not exist`
- 整个量化项目的 `big_train.py` 无法使用 GPU

**修复**（已执行）：

```bash
pip install --force-reinstall torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --no-deps torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**教训**：
1. 装任何「会拉 torch」的包之前，先 `pip install --dry-run` 看它要动什么
2. 微调环境应该建独立 venv（`python -m venv .venv-laya`），不要污染主环境
3. 远程机器上随便玩，本地主环境只放量化依赖

---

## 七、复现命令

```bash
# 1. 零样本验证（约 20 秒，含模型加载）
python scripts/test_laya.py

# 2. 构造示例数据
python -c "..."   # 见 data/laya/demo_train.csv

# 3. dry-run 检查数据格式
python -m laya.train_cli --data data/laya/demo_train.csv --dry-run \
  --question-id sentiment --instructions "判断这条A股公告对股价短期的影响方向"

# 4. 微调（4060 用 freeze-encoder，4090 可去掉）
python -m laya.train_cli --data data/laya/demo_train.csv \
  --eval data/laya/demo_eval.csv --out runs/laya_head \
  --question-id sentiment --instructions "判断这条A股公告对股价短期的影响方向" \
  --epochs 10 --micro-batch 4 --grad-accum 2 --freeze-encoder --device cuda
```
