# 图表规范审计（第 14 条）

> 2026-11-13｜要求：中文坐标与单位、图题、数据来源、避免双纵轴、色盲可读

## 一、先立规矩：`scripts/figstyle.py`

所有新图统一走这个模块，四项要求一次性满足：

| 要求 | 实现 |
|---|---|
| 中文坐标与单位 | `font.sans-serif='Microsoft YaHei'` + 轴标签强制带单位（如「净值（倍，对数坐标）」） |
| 图题 | 每图必须有 `set_title`，写清「是什么 + 区间」 |
| 数据来源 | `save(fig, path, note='数据来源：…')` 在图底加灰色小字脚注 |
| 避免双纵轴 | 禁用 `twinx()`；两个量纲就分两个子图（A1A2 图就是三行子图，不是双轴） |
| 色盲可读 | Okabe-Ito 8 色板（`PALETTE`），不用 jet/tab10 |

## 二、已规范化（15 张）

| 图 | 状态 | 做了什么 |
|---|---|---|
| `cluster_trio.png` | ✅ | 轴改中文带单位（聚类数 k（类）/ 簇内平方和（无量纲）/ 轮廓系数（0~1）/ ARI（0~1）），图题全中文，底部加来源 |
| `shap_global.png` | ✅ | x 轴改「平均 \|SHAP\| 值（无量纲）」，底部加来源（模型 + 抽样说明） |
| `tsne_factors.png` | ✅ | 图题中文，底部加来源（3,000 只 × 42 因子、perplexity=30） |
| `A1A2_v2.png` | ✅ | 纵轴补「累计净值（倍，对数坐标）」+「年份」轴名，底部加来源，并说明「等距=等比例涨跌」 |
| **9 张事件研究图** | ✅ | `event_study.py` 的 `plot_car()` 接入 figstyle：轴名「事件后窗口（交易日，对数坐标）」「累计超额收益 CAR（%）」，配色换 Okabe-Ito，底部自动写「事件表名 + 事件数 + 基准」 |

## 三、待处理（10 张）

| 图 | 生成脚本 | 已知问题 | 优先级 |
|---|---|---|---|
| `A1A2_combo.png` | `combine_v2.py` | 同 A1A2_v2，未补来源 | 高（报告主图） |
| `big_eval.png` | `big_eval.py` | ✅ 已规范化：中文轴带单位（多空组合净值（倍）/日期（年）/夏普比率（无量纲）/模型），Okabe-Ito 配色，底部来源 | — |
| `factor_scorecard.png` | `scorecard_plot.py` | ✅ 已规范化：IC 轴标「无量纲」、PCA 轴标「（比例，0~1）/（个）」，副标题写区间，底部来源 | — |
| `ev_affair.png` / `ev_demo_cn.png` / `ev_demo_us.png` / `ev_earnings.png` / `ev_earnings_lag1.png` / `ev_margin.png` / `ev_st_add.png` / `ev_st_remove.png` / `ev_yjyg.png`（9 张） | `event_study.py` | ✅ **已规范化**：统一走 `figstyle.save`，中文轴名带单位、Okabe-Ito 配色、底部数据来源（含事件表名 + 事件数 + 基准） | — |
| `taa_v3.png` / `regime_rotation.png` / `multifactor_nav.png` / `muon_ab.png` / `svm_*.png` / `macro_switch.png` / `tree_pruning.png` | 各自脚本 | 早期图，未审 | 低（多为"证伪路径"，报告里不主推） |

**已完成的一处改动**：`event_study.py` 的 `plot_car()` 接上 figstyle，一次性规范了 9 张事件图（比预估的 7 张多两张：`ev_st_add`、`ev_yjyg`）。

## 四、`combine_v2.py` 与 `A1A2_v2` 的关系

`A1A2_combo.png` 和 `A1A2_v2.png` 内容重叠（同一批净值，不同排版），报告里只该留一张。
建议：保留 `A1A2_v2.png`（已规范化），`A1A2_combo.png` 归档不再引用。

## 五、检查清单（以后每张新图过一遍）

- [ ] 轴标签带中文 + 单位？
- [ ] 图题写清「是什么 + 区间」？
- [ ] 底部有「数据来源：…」？
- [ ] 没有 `twinx()` 双纵轴？
- [ ] 用的不是红绿对比色（用 `PALETTE`）？
- [ ] 对数轴有标注？
