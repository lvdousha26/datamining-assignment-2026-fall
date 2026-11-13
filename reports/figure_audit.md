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

## 二、已规范化（4 张）

| 图 | 状态 | 做了什么 |
|---|---|---|
| `cluster_trio.png` | ✅ | 轴改中文带单位（聚类数 k（类）/ 簇内平方和（无量纲）/ 轮廓系数（0~1）/ ARI（0~1）），图题全中文，底部加来源 |
| `shap_global.png` | ✅ | x 轴改「平均 \|SHAP\| 值（无量纲）」，底部加来源（模型 + 抽样说明） |
| `tsne_factors.png` | ✅ | 图题中文，底部加来源（3,000 只 × 42 因子、perplexity=30） |
| `A1A2_v2.png` | ✅ | 纵轴补「累计净值（倍，对数坐标）」+「年份」轴名，底部加来源，并说明「等距=等比例涨跌」 |

## 三、待处理（21 张）

| 图 | 生成脚本 | 已知问题 | 优先级 |
|---|---|---|---|
| `A1A2_combo.png` | `combine_v2.py` | 同 A1A2_v2，未补来源 | 高（报告主图） |
| `big_eval.png` | `big_eval.py` | 疑有英文轴、缺来源 | 高 |
| `factor_scorecard.png` | `factor_scorecard.py` | 42 因子条形图，缺来源 | 高 |
| `ev_margin.png` / `ev_st_add.png` / `ev_st_remove.png` / `ev_yjyg.png` / `ev_earnings*.png` / `ev_demo_*.png` / `ev_affair.png` | `event_study.py` | 轴为中文但缺来源；`event_study.py` 内联 `plt.savefig`，未接 figstyle | 中（7 张同源，改一处即可） |
| `taa_v3.png` / `regime_rotation.png` / `multifactor_nav.png` / `muon_ab.png` / `svm_*.png` / `macro_switch.png` / `tree_pruning.png` | 各自脚本 | 早期图，未审 | 低（多为"证伪路径"，报告里不主推） |

**最高性价比的一处改动**：`event_study.py` 里有 7 张图都是它画的，接上 `figstyle.save(..., note=...)` 一次能规范 7 张。约 20 分钟。

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
