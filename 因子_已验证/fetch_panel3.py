# -*- coding: utf-8 -*-
"""面板 v3：新浪行情中心列表（市值/流通市值/PE/PB）+ 日线，扩池到市值前2000"""
import json, time, pickle, subprocess
from concurrent.futures import ThreadPoolExecutor

H = ['-H', 'User-Agent: Mozilla/5.0', '-H', 'Referer: https://vip.stock.finance.sina.com.cn/']
def curl(url, t=25):
    try:
        return subprocess.run(['curl', '-s', '--max-time', str(t)] + H + [url],
                              capture_output=True, timeout=t + 5).stdout.decode('utf-8', 'ignore')
    except Exception:
        return ''

rows = []
for page in range(1, 75):
    url = ('https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData'
           '?page=%d&num=80&sort=symbol&asc=1&node=hs_a' % page)
    txt = curl(url)
    try:
        d = json.loads(txt)
        if not d: break
    except Exception:
        time.sleep(0.5); continue
    for x in d:
        sym, name = x['symbol'], x['name']
        if sym.startswith('bj') or 'ST' in name or '退' in name: continue
        code = sym[2:]
        if not (code.startswith('60') or code.startswith('00') or code.startswith('30') or code.startswith('68')): continue
        try:
            rows.append((code, name, float(x['mktcap'] or 0) * 1e4, float(x['nmc'] or 0) * 1e4,
                         float(x['trade'] or 0), float(x['per'] or 0), float(x['pb'] or 0)))
        except Exception: pass
    if len(d) < 80: break

print('列表 %d 只' % len(rows))
rows.sort(key=lambda r: -r[2])
rows = rows[:2200]          # 市值前2200（覆盖绝大部分流动性）
print('取市值前 %d 只，市值门槛 %.1f 亿' % (len(rows), rows[-1][2] / 1e8))

def one(item):
    code, name, mcap, nmc, price, pe, pb = item
    sym = ('sh' if code.startswith('6') else 'sz') + code
    url = ('https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
           '?symbol=%s&scale=240&ma=no&datalen=2000' % sym)
    for _ in range(2):
        try:
            d = json.loads(curl(url, 20))
            if not d: return None
            return (code, name, mcap, nmc, price, pe, pb,
                    [(r['day'], float(r['close']), float(r['volume'])) for r in d])
        except Exception:
            time.sleep(0.15)
    return None

t0 = time.time(); panel = {}
with ThreadPoolExecutor(max_workers=26) as ex:
    for i, r in enumerate(ex.map(one, rows)):
        if r and len(r[7]) >= 500:
            panel[r[0]] = {'name': r[1], 'mcap': r[2], 'nmc': r[3], 'price0': r[4],
                           'pe': r[5], 'pb': r[6], 'k': r[7]}
        if (i + 1) % 500 == 0:
            print('  %d/%d 成功%d %.0fs' % (i + 1, len(rows), len(panel), time.time() - t0))

pickle.dump(panel, open('a_panel3.pkl', 'wb'))
print('完成 %d 只 %.0fs' % (len(panel), time.time() - t0))
