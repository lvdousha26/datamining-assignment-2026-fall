# -*- coding: utf-8 -*-
"""验证 Laya 多语言模型能否加载并做中文决策（不训练，只推理）"""
import sys, os, time
if __name__ == '__main__':
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from runlog import RunLogger
        RunLogger(__file__).__enter__()
    except Exception:
        pass

import laya

t0 = time.time()
print('加载 convaiinnovations/laya (multilingual)…', flush=True)
ag = laya.Agent(model_id_or_path='convaiinnovations/laya', subfolder='multilingual', device='cuda')
print(f'加载完成 {time.time()-t0:.1f}s | backend={ag.backend}', flush=True)

QS = {"sentiment": {
    "type": "choice",
    "instructions": "判断这条 A 股公告对股价短期的影响方向",
    "criteria": {
        "利好": "业绩预增、回购、中标、增持、分红",
        "中性": "常规事项、人事变动、章程修订",
        "利空": "股东减持、业绩预亏、诉讼、问询函、违规处罚",
    }}}

cases = [
    "关于控股股东、实际控制人股份减持计划的预披露公告",
    "预计2024年度归属于上市公司股东的净利润同比增长120%至150%",
    "关于收到深圳证券交易所关注函的公告",
    "关于以集中竞价方式回购公司股份的进展公告",
    "关于变更公司注册地址的公告",
]

print('\n%-52s %s' % ('公告标题', '分布'), flush=True)
for c in cases:
    t = time.time()
    r = ag.predict(c, QS, lang='zh')
    print('%-52s %s  (%.0fms)' % (c[:50], r, (time.time()-t)*1000), flush=True)
