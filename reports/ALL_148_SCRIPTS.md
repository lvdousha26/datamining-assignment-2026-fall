# 148 个脚本全清单（逐条，无遗漏）

> 2026-11-06｜`D:\quant\scripts` 56 个 + `C:\Users\25097\WorkBuddy\2026-10-30-23-31-35` 92 个
> 状态：**A** 可实盘｜**B** 正超额但弱｜**C** 有信号不构成策略｜**D** 已证伪｜**E** 未验证｜**—** 非策略（工具）

---

# 一、BASE 目录 92 个（`C:\Users\25097\WorkBuddy\2026-10-30-23-31-35`）

## 1. 组合配置类（18）

| 脚本 | 内容 | 状态 / 数字 |
|---|---|---|
| `spec3.py` | **策略③** 定稿：红利低波50%+上证丹尼斯50% / 纳指动量 / 纳指MA125 / BRK / 黄金 / 国债 | **A** 11.08% / 夏普 0.93 / 回撤 −22.5% |
| `weight_search.py` | 9 腿权重网格（0.1 步长，单腿≤40%），最大夏普 + 分半 | B 策略③的构成来源 |
| `weight_search2.py` | 权重 v2：单腿≤25%、国债≤20%，20 年窗口验证 | B 约束版 |
| `joint_search.py` | 参数×权重联合搜索：600×2907=174 万组合，分半 OOS | B 策略③参数来源 |
| `multi.py` | 把最好的几个策略组合，测相关/组合夏普 | B |
| `combine.py` | 双层：截面选股(alpha)×时序择时(beta预算)，四组对照 | B |
| `rot_blend.py` | 跨资产轮动 vs 策略③：相关 + 卫星叠加 | D 轮动不敌③ |
| `asset_rot.py` | 7 资产截面动量 topN + 绝对动量过滤 | D 同 TAA 族 |
| `rotation.py` | 7 只 ETF 轮动回测 | D |
| `switch.py` | 模式切换（8 种模式对照） | D |
| `two.py` | 两资产切换对照 | D |
| `tgm.py` | 9 腿统一窗口（2007-2026）年化/夏普/Calmar/换手 | B 汇总工具 |
| `global.py` | 各市场：买入持有 vs 同一趋势策略 | D 结论：择时无效 |
| `improve.py` | 改进对比（多版本横向） | — 辅助 |
| `scan2020.py` | 各策略族参数扫描（2020-2026） | — 辅助 |
| `recent.py` | 2020-2026 近 6.8 年表现复核 | — 辅助 |
| `trainset.py` | 训练集起点×长度对关键年份的影响 | — 稳健性 |
| `param50k.py` | 解释 k 参数（偏离多少到满仓/空仓） | — 说明 |

## 2. 趋势线动量择时族（13）

`bt1`~`bt9` 是同一族的迭代：**趋势线动量（k=5~8，窗口 252）在上证/标普/纳指/恒生上的参数敏感性、口径对照、扩展窗口、绝对价格、稳健性**。

| 脚本 | 内容 | 状态 |
|---|---|---|
| `bt1.py` | 主策略 k=5 / 窗口 252 / 成本 5bp | D 超额 t 不达标 |
| `bt2.py` | k 敏感性（训练集滞后 252 天，1996 起） | D |
| `bt3.py` | 三种口径对照 | D |
| `bt4.py` | 主口径结果 | D |
| `bt6.py` | 变体对比（年化/夏普/Calmar/换手） | D |
| `bt7.py` | 变体对比 v2 | D |
| `bt8.py` | 扩展窗口（1990→前一年），测试 2020-2026 | D |
| `bt9.py` | 绝对价格扩展窗口，1996-2026 | D |
| `momentum.py` | 方向×市场：sign=+1 动量 / −1 均值回归 | D 两方向都不行 |
| `ma_struct.py` | MA 腿两个结构改法：缓冲带 vs 双均线 | D |
| `classic.py` | CAPE + A股PE 估值 | C |
| `capacity.py` | 换手结构（1996-2026） | — 容量 |
| `factor.py` | 市场因子暴露（日频） | — 归因 |

**记分卡实测（`scorecard.py`，2007-02~2026-10）**：纳指趋势线 k=8 10.73%/夏普0.50/**超额t −3.30**；上证 line 7.21%/0.34/**t +3.23**；上证 don 7.00%/0.39/t +2.67；沪深300 格雷厄姆PE 8.02%/0.30/**t +3.24**；恒生趋势线反转 4.86%/0.16/**t +3.96**；标普丹尼斯55/20 2.39%/0.04/**t −7.94**；上证 macd 4.97%/0.19、grid 3.58%/0.14、ddt 2.24%/0.02。

## 3. 事件 / 特殊情形（20）

| 脚本 | 内容 | 状态 |
|---|---|---|
| `crash_buy.py` | 股灾抄底：99% 时间空仓等崩盘 | **E** |
| `crash_buy2.py` | 抄底 v2：带止盈/时限 | **E** |
| `crashes.py` | 历次大跌（跌>20%）策略在每段的真实表现 | — 归因 |
| `bottom_signal.py` | 大跌到底 + 横盘三月 → 进攻（四组对照） | **E** |
| `bottom_anatomy.py` | 逐点解剖历史大底，信号抓到了吗、已涨多少 | **E** |
| `euphoria.py` | 狂热指标读数：历次顶部 vs 平时 | **E** |
| `dennis_snipe.py` | 丹尼斯 90/40 抢跑：δ 以内→突破率/前向收益 | **E** |
| `snipe_quant.py` | 狙击量化：将破未破时买入 | **E** |
| `snipe_walkforward.py` | 抢跑 walk-forward（过去24月选参→后12月交易） | **E** |
| `gold_btc.py` | 黄金/BTC/新兴市场：单项 + 择时 + 组合 | **E** 样本短 |
| `oil_carry.py` | 原油期限结构 carry（backwardation 做多） | **E** 仅 6 年 |
| `oil_strategy.py` | 原油：趋势 + 曲线形态 + carry 过滤 | **E** 仅 6 年 |
| `brk.py` | 伯克希尔 / 段永平代理（苹果+茅台） | **E** |
| `season.py` | 月度季节性 / 年末年初 / 走前检验 | **E** |
| `accounts.py` | 新增投资者数与未来 6 月收益 | **E** |
| `real_econ.py` | 实物经济景气（沪铜/螺纹钢/BDI 同比）→ A股择时 | **C** |
| `inflation.py` | 通胀相关择时 | **E** |
| `prior_check.py` | 先验检验 1：月度收益分布 | — 检验 |
| `prior_check2.py` | 先验 2：牛市时间占比（三种定义） | — 检验 |
| `val_timing.py` | 估值分位择时（PE<15% 满仓 / >80% 减仓） | **B** 超额t +3.24 |

## 4. 因子 / 选股（12）

| 脚本 | 内容 | 状态 |
|---|---|---|
| `factor_lab.py` | 研报→假设→算子→因子实验室（基本/矩/相关/图/频谱/掩码） | D 大部分为噪声 |
| `factor_lab2.py` | 第二层：中性化 + 正交 + 合成，筛真因子 | D |
| `factor_lab3.py` | 复现卖方研报：隔夜-日内分离、RPV 四象限 | C 隔夜因子有信息 |
| `factor_x.py` | 横截面因子引擎 v2（跳跃检测修复不复权） | D |
| `pipeline_v2.py` | 扩池 2096 只 + 7 因子 + FM/FDR + buffer 组合 | D 幸存者偏差未除 |
| `strategy_design.py` | 多因子原型：正交→FM→组合优化→择时→多重检验 | D |
| `intraday_verify.py` | INTRADAY5：(close/open−1) 均值取低 20% | **C** |
| `quad_add.py` | QUAD_ASYM（低开高走−高开低走）作卫星入策略③ | **C** 已验证方向 |
| `quad_verify.py` | QUAD 严格复检（vs 池等权、分半 2019-22/2023-26） | **C** |
| `sw_mom.py` | 申万一级 28 行业 12-1 动量，等权 topN | **C** 数据源干净 |
| `sw_mom2.py` | 补充：行业反转/低波/风险调整动量 | **C** |
| `brain_target.py` | 构造 WorldQuant BRAIN 标准 alpha（Sharpe>1.25/Fitness>1.0） | **E** 未达成 |

## 5. 检验 / 诊断（12）

`pbo_test`（CSCV/PBO + Deflated Sharpe + 零假设最优夏普）｜`overfit`（IS/OOS 衰减）｜`overfit_perturb`（参数扰动分布）｜`utility`（CRRA 确定性等价 + 块自举）｜`scorecard`（**统一记分卡，所有策略汇总**）｜`eval_s3`（策略③机构视角体检 + 多资产 beta 归因）｜`evaluate_live`（模拟盘实际 vs 回测）｜`live_signal`（策略③月度执行信号）｜`rmb_grid`（美元 vs 人民币口径选参）｜`fx_compare`（人民币口径 vs 各资产）｜`spx_test`（纳指 vs 标普）｜`us_compare`（中美对比）

全部为工具/检验，**无独立策略**。

## 6. 调参（5）

`tune`（参数网格→候选池→前向贪心）｜`tune2`（局部搜索 + 邻近平坦性 + 2020后）｜`tune3`（加入经典策略：200日均线/布林带/双动量）｜`tune4`（第四轮验收 + 对比图）｜`tune5`（全因子网格 2250 组，修复丹尼斯下轨清仓）

→ **是策略③参数体系的来源，本身不是策略**。

## 7. 绘图（9）

`plot` ~ `plot9`：净值/回撤/分年度/最优组合对比图。**无策略**。

## 8. 抓数（3）

`fetch_panel` / `fetch_panel2` / `fetch_panel3`：全 A 面板 v1/v2/v3（市值/流通市值/PE/PB + 日线）

---

# 二、QUANT 目录 56 个（`D:\quant\scripts`）

## 1. 个股模型（12）

| 脚本 | 内容 | 状态 / 数字 |
|---|---|---|
| `big_train.py` | **量价主模型**：LGBM + Transformer(3seed) + Ridge，42 因子 | **B** IC 0.1044 |
| `big_models.py` / `big_models_v0.py` | 旧版训练（v0 为归档） | — 历史 |
| `purged_cv.py` | 5 折 Purged+Embargo CV | — 检验 |
| `purged_cv_fin.py` | 同上 + 21 个基本面因子 | **D** 4/5 折变差 |
| `svm_daily.py` | SVM 日频选股（C/γ 网格 + 安慰剂） | **D** |
| `svm_cs.py` | SVM 截面（换目标：排名/分类） | **D** |
| `svm_stack.py` | SVM 残差叠加到 OLS | **D** |
| `tree_model.py` | 决策树 + 剪枝 | **D** 0.0714 ≈ OLS < 等权 0.0862 |
| `multifactor.py` | 5 种合成 + 周频 top20% + rank buffer | **D** OOS 弱 |
| `p6_combo.py` | P6-lite：8 因子等权 rank 合成 + K=5 分组 | **D** |
| `mine_factors.py` | P1 因子生成（14 个候选） | — 数据 |
| `live_screen.py` | **禁买清单生成器**（模型冻结） | **B** +2.87% / 夏普 0.53 |

## 2. 因子 / 审计（8）

`factor_scorecard`（42 因子 8 项打分卡）｜`scorecard_plot`（四宫格）｜`audit_data`（inf 污染审计）｜`audit_overfit`（IS/OOS 衰减 + 安慰剂）｜`audit_redundancy`（有效检验数 19.6）｜`pv_deep`（量价专项：逐年/分位/可交易性/D1画像）｜`fin_ic_probe`（21 基本面因子 IC）｜`big_eval`（小市值中性化/流动性分层/成本敏感）

## 3. 行业（5）

`ind_rolling`（point-in-time 聚类 31 组）｜`ind_check`（行业中性化检验）｜`swap_industry`（申万二级替换）｜`apply_sw_industry`（写入 arr.npz）｜`hot_sector_backtest`（**热点行业动量复现**，D：0成本 −5.88%）

## 4. 数据（9）

`big_download`（全A 5378 只含退市）｜`big_features`（42 因子面板）｜`big_fin_download`（216 期财报）｜`big_fin_features`（21 个 PIT 因子 + 申万行业）｜`download_data`（新浪源抽样）｜`st_download` / `st_finalize` / `st_merge` / `st_sensitivity`（时变 ST 日历）

## 5. 策略落地（12）

| 脚本 | 内容 | 状态 |
|---|---|---|
| `long_strategy.py` | **质量+估值长线**（季度调仓 top300） | **A** +9.92% / 夏普 1.03 |
| `strategy_compare.py` | 全市场 vs 行业内剔底（同口径对照） | **B** +6.55% / +5.57% / +2.86% |
| `taa_v3.py` | **TAA v3**：ERP 择时 + 精简池 | **B** 6.79% / 0.49 |
| `regime_rotation.py` | TAA v2：动量广度 + 风险预算 | **D** 0.15~0.25 |
| `lockbox_eval.py` | 一次性封存验证（runs=1） | — 检验 |
| `rebuild_ridge.py` | 只重建 Ridge 元模型（不重训） | — 工具 |
| `stack_ridge.py` | Ridge 元模型模块 | — 工具 |
| `ridge_check.py` | OLS vs Ridge 等价性/增益检验 | — 检验 |
| `turnover_scan.py` | 换手-超额前沿（周期×buffer） | — 参数 |
| `delist_stress.py` | 退市风险压力 + 流动性过滤 | — 检验 |
| `macro_switch.py` | 信用利差 + 股债相关性开关 | **C** 仅信用利差 +0.12 夏普 |
| `remote.py` | paramiko 远程执行/SFTP | — 工具 |

## 6. 实验 / 对照（8）

`tf_cv_probe`（TF 2 折探针 + arch 消融）｜`lgbm_bag_probe`（bagging 归因）｜`muon_ab`（Muon vs AdamW）｜`muon_optimizer`（NS 正交化）｜`muon_plot`｜`adamw_lr_control`（lr 对照）｜`tf_net`（网络定义）｜`checkpoint`（断点保存）

---

# 三、按"能不能实盘"排序

| 排名 | 策略 | 脚本 | 数字 | 等级 |
|---|---|---|---|---|
| 1 | **策略③** | `spec3.py` | 11.08% / 夏普 0.93 / 回撤 −22.5% | **A** |
| 2 | **质量+估值长线** | `long_strategy.py` | +9.92% / 1.03 / −12.2% | **A** |
| 3 | 禁买清单（量价） | `live_screen.py` | +2.87% / 0.53 / 换手 5.4% | **B** |
| 4 | 行业内剔底·聚类 | `strategy_compare.py` | +5.57% / 5.16 | **B** |
| 5 | 格雷厄姆 PE 择时 | `val_timing.py` | 超额 t +3.24 | **B** |
| 6 | 上证 line / 恒生反转 | `bt*.py` | 超额 t +3.23 / +3.96 | **B** |
| 7 | TAA v3 | `taa_v3.py` | 6.79% / 0.49（不互补） | **B** |
| 8 | 其余 133 个 | — | 0 或负 | C/D/E/— |

**统计**：A 级 2 个｜B 级 7 个｜C 级 8 个｜D 级 16 个（含 6 个单指数择时）｜E 级 15 个｜其余 100 个为工具/数据/绘图/历史版本。
