# 报告索引（reports/ 里什么值得看）

> 2026-11-13 整理｜共 50+ 份报告，按主题分组。**加粗 = 论文/答辩要引用的核心结论**

## 一、主结论

| 文件 | 内容 | 一句话 |
|---|---|---|
| **`big_report.md`** | 全流程主报告：面板、模型、CV、落地 | 5 折 CV IC 0.1041±0.0094，唯一可落地形态是禁买清单 |
| **`live_strategies.md`** | 当前可执行清单 | 全市场剔底 10%，预期 +2.5~3%/年 |
| **`STRATEGY_MANUAL.md`** | 规则级策略手册 | 只跑 A1（策略③），A2 最优权重 0% |
| `data_fix_v2_report.md` | 数据修复（hfq/断层）与重算 | 修复后四行数字各上修 0.8pp |
| `deployment_plan.md` | 部署方案 | ETF 主线 80–90%，个股只做减法 |

## 二、被证伪的路径（负结果，答辩重点）

| 文件 | 路径 | 结论 |
|---|---|---|
| `svm_report.md` / `svm_cs_report.md` | SVM 日频 / 截面 | 无增量 |
| `short_b_review.md` | 多空组合 | 封存段 −1.08% |
| `hot_sector_review.md` | 行业动量追涨 | 0 成本 −5.88%/年 |
| `fin_factor_report.md` | 基本面因子加成 | 5 折 4 折变差 |
| `muon_report.md` | Muon 优化器 | 与 AdamW 持平 |
| `taa_v3_report.md` | TAA 择时 | 夏普 0.49，打不过策略③ |
| `strategy_rotation_report.md` | 策略轮动 12 变体 | 全部跑输固定配置 |
| `ev_yjyg.md` | 业绩预告事件 | 5 日 CAR −0.58%，t −0.83 |

## 三、本轮新增（11-09 ~ 11-13）

| 文件 | 内容 | 关键数字 |
|---|---|---|
| **`text_bench.md`** | 五路文本模型横向 | 表格 93.2% > BERT 89.2% > LR 88.2% > 正则 84.6% |
| **`return_label.md`** | 换标签（次日超额） | 全部模型 34~36%，一致率仅 35.7% |
| `laya_yjyg.md` | Laya 冻结微调 | 60.8% → 78.7%（63 秒） |
| **`cluster_trio.md`** | 聚类三件套 | 肘部 k=4，ARI@31 仅 0.392 |
| `shap_explain.md` | SHAP 全局 + 3 样本 | size 11.6% / turn5 10.3% |
| `bonus_d.md` | 不平衡/交互/t-SNE/特征选择 | 类别权重 macro-F1 0.891；42→15 维 AUC 仅掉 0.003 |
| `cls_bottom.md` | 分类化（垫底 20%） | PR-AUC 0.365 vs 基线 0.200，t=12.63 |
| `wq101.md` | WQ101 因子复现 | 价量背离 IC 0.051，九成被现有 pvcorr20 解释 |
| `chan_factors.md` | 缠论结构因子（分型/笔/中枢/背驰） | 笔与中枢宽度是 `dev20`/`std60` 换皮；**中枢位置残差 IC +0.0115、9 年全正、Q5−Q1 +15.2%**；背驰控制超跌后 9 年全负 |
| `neutral_compare.md` | 行业中性化分组 | 申万一级让 81% 因子 IC 下降 |
| `nn_loss.md` | 神经网络损失对比 | rank 回归 0.112 > pairwise 0.108 > BCE 0.084 |
| `zt_factor.md` | 涨停封单因子 | 高封单组次日 +1.48%，但 34% 是一字板买不进 |
| `ev_st_remove.md` | ST 摘帽事件 | 250 日 CAR −13%，安慰剂分位 40% → 非事件效应 |

## 四、方法论文档（写"方法论"章节时抄这些）

| 文件 | 内容 |
|---|---|
| `event_study_guide.md` | 事件研究框架使用说明（CAR/安慰剂/同侪三件套） |
| `data_dictionary.md` | 110 个字段口径 |
| `factor_quality_review.md` / `factor_scorecard.csv` | 42 因子打分卡 |
| `group_division.md` | 分工表模板（待填姓名） |

## 五、怎么快速判断一份报告可不可信

1. **看有没有安慰剂/基线**：只有 IC/AUC 没有对照的报告，不作为结论引用；
2. **看样本量**：涨停封单只有 13 天、ST 摘帽只有 260 个事件——都标了"描述性统计"；
3. **看是否碰了封存段**：`lockbox_state.json` 里 `runs=1`，任何 2025-07 之后的"验证"都作废。
