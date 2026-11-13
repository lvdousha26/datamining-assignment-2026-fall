# 数据采集存档（谁抓的、怎么抓的、什么时候抓的）

> 2026-11-13｜作业要求（第 15 条）：采集脚本或查询语句 + 采集时间一并存档
>
> 本文件由 `scripts/make_provenance.py` 从 `logs/index.csv` 与文件修改时间自动生成，
> 每次跑数据脚本后重跑一次即可更新。**不要手改**。

## 一、逐数据集

| 数据集 | 落盘路径 | 采集脚本 | 接口 / 查询 | 规模 | 最近采集时间 | 覆盖区间 |
|---|---|---|---|---|---|---|
| 后复权日线 | `data/big/raw_hfq/*.parquet` | `scripts/big_download_hfq.py` | 东方财富 `stock_zh_a_hist(adjust='hfq')` | 5,446 只 | 2026-11-09 07:46 ⚠️仅文件时间 | 2016-01 ~ 2026-11 |
| 日收益矩阵 | `data/big/daily_ret.npz` | `scripts/market_verify.py` | 由 raw_hfq 计算（去停牌/断层/异常） | 2,622 日 × 5,342 只 | 2026-11-09 07:5x ⚠️仅文件时间 | 2016-01-12 ~ 2026-10-30 |
| 因子面板 | `data/big/arr.npz` | `scripts/big_features.py` | 由 daily_ret + raw_hfq 计算 | 2,622 × 5,342 × 42 | 2026-11-09 08:26 ⚠️仅文件时间 | 同上 |
| 申万行业（二代） | `data/big/sw_industry.parquet` | `scripts/fetch_sw.py` | 申万宏源官网 `index_component_sw` | 59,467 行 / 128 类 | 2026-11-06 ⚠️仅文件时间 | 2016 ~ 2026（按年） |
| 申万一级映射 | `data/big/ind_map.parquet` | `scripts/fetch_sw.py` | 同上（一级聚合） | 5,221 只 / 31 类 | 2026-11-06 ⚠️仅文件时间 | 当前成分（无历史） |
| 业绩预告 | `data/big/yjyg/*.parquet` | `scripts/fetch_yjyg.py` | 东方财富 `stock_yjyg_em` | 235,229 条 / 45 期 | 2026-11-09 09:38 ⚠️仅文件时间 | 2016Q1 ~ 2026Q3 |
| ST 日历 | `data/big/st_calendar.parquet` | `scripts/fetch_st.py` | baostock `query_history_k_data_plus` 的 `isST` | 151,170 行 / 322 只 | 2026-11-09 ⚠️仅文件时间 | 2015-01 ~ 2026-05 |
| **两融标的名单** | `data/big/margin/*.parquet` | `scripts/fetch_margin.py` | 深交所 `stock_margin_underlying_info_szse`、上交所 `stock_margin_detail_sse` | 1,897 日快照 | 2026-11-13 22:49（有日志） | 2019-01 ~ 2026-10 |
| **两融融资余额** | `data/big/margin_bal/*.parquet` | `scripts/fetch_margin_bal.py` | 同上（取 `融资余额` 列） | 380 个抽样日 | 2026-11-13 23:16（有日志） | 2019-01 ~ 2026-10（每 5 日） |
| 监管类公告 | `data/big/notice_reg/*.parquet` | `scripts/fetch_notice_reg.py` | 东财 `stock_notice_report`（关键词过滤） | 54 个日期 | 2026-11-13 23:26（有日志） | **仅最近约 1 个月**（接口限制） |
| 美股后复权 | `data/us/*_hfq.parquet` | `scripts/affair_factor.py` 的 `load()` | akshare `stock_us_hist(adjust='hfq')` | 11 只 | 2026-11-10 23:22（有日志） | 2004 ~ 2026 |

## 二、为什么有的采集时间只能给「最近一次」

`logs/index.csv` 只记录**跑过的运行**（2026-11-09 之后接入 runlog）。
更早的采集（`big_download_hfq` 等）没有运行日志，只能用文件 mtime 近似——这就是第 15 条要补的东西，
从现在起所有新脚本都走 `runlog.py`，每次运行写 `logs/<时间戳>_<脚本>.json`（含 argv、耗时、状态）。

## 三、复现命令

```bash
python scripts/big_download_hfq.py      # 行情（断点续传）
python scripts/market_verify.py         # 清洗 → daily_ret.npz
python scripts/big_features.py          # 42 因子面板（约 40 分钟）
python scripts/fetch_yjyg.py            # 业绩预告
python scripts/fetch_st.py              # ST 日历
python scripts/fetch_margin.py --start 2019-01-01 --workers 8     # 两融标的（约 11 分钟）
python scripts/fetch_margin_bal.py --step 5                        # 融资余额（约 3 分钟）
```

## 四、已知的数据时效与限制

| 数据集 | 时效 | 限制 |
|---|---|---|
| 行情 | T+1（收盘后） | 东财 A 股需 `hfq`，`qfq` 是减法复权会出现负价 |
| 业绩预告 | 实时 | 无限制 |
| ST 日历 | 日更 | baostock 只精确覆盖 322 只 |
| 两融名单 | T+1 | 上交所口径是"当日有融资余额" |
| 监管公告 | T+1 | **只保留最近约 1 个月，无法回补历史** |
| 美股 | 实时 | 需外网 |
