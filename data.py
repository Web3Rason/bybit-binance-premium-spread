# -*- coding: utf-8 -*-
"""
資料層 — Binance USDⓈ-M 永續 與 Bybit V5 線性永續 的原生 REST 歷史抓取。

研究核心要對齊比較「同一合約在兩家的溢價/標記/指數/資費」，所以一律用
交易所『原生』 endpoint（禁用 ccxt）。各 endpoint 回傳結構皆已實測（2026-06-16）。

K 線統一回 pandas DataFrame，欄位：time(ms,int 升序)、close(float)。
（mark/index/premium 的 K 線只有 OHLC 沒有量，研究只取 close 收盤值。
  last price K 線另含 volume，需要時用 kind="last" 並讀 volume 欄。）

歷史深度（1m，實測）：
  Binance premiumIndexKlines 3年+ / klines(last) 4年+
  Bybit  premium-index-price-kline 2年+ / kline(last) 3年+
  → 取交集，premium 對齊軸約 2 年。

官方文件：
  Binance: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data
  Bybit V5: https://bybit-exchange.github.io/docs/v5/market/kline
"""
import os
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

import pandas as pd

BINANCE = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

MIN_MS = 60_000

# Binance K 線各「種類」對應 endpoint 與 symbol 參數名
#   index 用 pair 參數（其餘用 symbol），這是實測確認的差異。
_BN_KLINE = {
    "last":    ("/fapi/v1/klines", "symbol"),
    "mark":    ("/fapi/v1/markPriceKlines", "symbol"),
    "premium": ("/fapi/v1/premiumIndexKlines", "symbol"),
    "index":   ("/fapi/v1/indexPriceKlines", "pair"),
}
# Bybit K 線各「種類」對應 endpoint（皆用 symbol，category=linear）
_BB_KLINE = {
    "last":    "/v5/market/kline",
    "mark":    "/v5/market/mark-price-kline",
    "premium": "/v5/market/premium-index-price-kline",
    "index":   "/v5/market/index-price-kline",
}


# ───────────────────────── HTTP（含退避 + weight 監控） ─────────────────────────
# Binance REQUEST_WEIGHT 上限 2400/分/IP。主動讀 used-weight 動態降速，永遠不撞 418。
_BN_WEIGHT_SOFT = 1800   # 超過此值：額外降速
_BN_WEIGHT_HARD = 2200   # 逼近上限：停手等到下一分鐘 weight 重置


def _throttle_binance(used_weight_hdr):
    """讀 X-MBX-USED-WEIGHT-1M 主動降速（防 418 IP ban 的最關鍵防護）。"""
    if not used_weight_hdr:
        return
    try:
        used = int(used_weight_hdr)
    except (ValueError, TypeError):
        return
    if used > _BN_WEIGHT_HARD:
        time.sleep(61 - (time.time() % 60))  # 等到下一個整分鐘重置
    elif used > _BN_WEIGHT_SOFT:
        time.sleep(2)


def _get(base: str, path: str, params: dict, retries: int = 5):
    """GET JSON。Binance：讀 used-weight 主動降速、418 IP ban 直接中止；
    Bybit：403 限速封鎖（約10分鐘）直接中止；兩家 429 指數退避（尊重 Retry-After）。"""
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                body = json.loads(r.read())
                if base == BINANCE:
                    _throttle_binance(r.headers.get("X-MBX-USED-WEIGHT-1M"))
                return body
        except urllib.error.HTTPError as e:
            msg = (e.read()[:160].decode("utf-8", "ignore")) if hasattr(e, "read") else ""
            if e.code == 418:
                raise RuntimeError(f"Binance IP banned (418)，請停手等待解封：{msg}")
            if e.code == 403:  # Bybit access too frequent，封約 10 分鐘，狂撞只會更慘
                raise RuntimeError(
                    f"Bybit 403 限速封鎖（約10分鐘），請降低節奏後重跑：{msg}")
            if e.code == 429:
                wait = max(int(e.headers.get("Retry-After", 0)), min(2 ** attempt, 60))
                time.sleep(wait + 1)
                last_err = e
                continue
            last_err = e
            time.sleep(min(2 ** attempt, 60))
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"GET 失敗 {path}: {last_err}")


# ───────────────────────── 合約清單 / 規格 ─────────────────────────
def binance_perps() -> set[str]:
    """所有 TRADING 中的 Binance USDT 永續合約代號。"""
    info = _get(BINANCE, "/fapi/v1/exchangeInfo", {})
    return {
        s["symbol"] for s in info.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
    }


def bybit_instruments() -> dict[str, dict]:
    """全市場 Bybit 線性 USDT 永續：{symbol: {interval_min, upper, lower}}。
    upper/lower = 資費上下限（cap，可能不對稱），interval_min = 結算週期(分鐘)。
    cursor 分頁。"""
    out = {}
    cursor = ""
    while True:
        params = {"category": "linear", "limit": "1000"}
        if cursor:
            params["cursor"] = cursor
        d = _get(BYBIT, "/v5/market/instruments-info", params)
        result = d.get("result") or {}
        for it in result.get("list") or []:
            sym = it.get("symbol", "")
            if not sym.endswith("USDT") or it.get("status") != "Trading":
                continue
            if it.get("contractType") != "LinearPerpetual":
                continue
            out[sym] = {
                "interval_min": int(it.get("fundingInterval") or 480),
                "upper": float(it.get("upperFundingRate") or 0) or None,
                "lower": float(it.get("lowerFundingRate") or 0) or None,
            }
        cursor = result.get("nextPageCursor") or ""
        if not cursor:
            break
    return out


def common_symbols() -> list[str]:
    """兩家共同上市的 USDT 永續代號（取交集，不做成交額篩選）。"""
    bn = binance_perps()
    bb = set(bybit_instruments().keys())
    return sorted(bn & bb)


# Binance 未列於 fundingInfo 的合約採用平台預設（官方：cap ±2%、結算 8h）
BINANCE_DEFAULT_CAP = 0.02
BINANCE_DEFAULT_INTERVAL_H = 8


def binance_funding_info() -> dict[str, dict]:
    """Binance 各合約『當前』資費上下限與結算週期：
    {symbol: {cap, floor, interval_h}}。
    僅含有特殊設定者（實測 667 檔涵蓋多數活躍合約）；缺漏者由呼叫端套預設。
    注意：此為當前值，歷史 cap 可能不同（公開 API 不提供歷史 cap，分析需註明）。"""
    fi = _get(BINANCE, "/fapi/v1/fundingInfo", {})
    out = {}
    for x in fi:
        out[x["symbol"]] = {
            "cap": float(x["adjustedFundingRateCap"]),
            "floor": float(x["adjustedFundingRateFloor"]),
            "interval_h": int(x["fundingIntervalHours"]),
        }
    return out


def infer_interval_h(funding_times_ms: list[int]) -> float | None:
    """從歷史結算時間戳推算『當時實際』結算週期(小時)。
    取相鄰間隔的眾數，避免被偶發缺漏/週期切換干擾。極端期週期會縮短(8h→4h→1h)，
    用此函式才能反映事件當下的真實週期，而非當前 instruments 設定。"""
    ts = sorted(set(int(t) for t in funding_times_ms))
    if len(ts) < 2:
        return None
    gaps = [round((ts[i + 1] - ts[i]) / 3_600_000) for i in range(len(ts) - 1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return None
    return max(set(gaps), key=gaps.count)


# ───────────────────────── K 線抓取 ─────────────────────────
def _empty_kline(with_vol: bool) -> pd.DataFrame:
    cols = ["time", "open", "high", "low", "close"] + (["volume"] if with_vol else [])
    return pd.DataFrame(columns=cols)


def fetch_binance_kline(symbol: str, kind: str, start_ms: int, end_ms: int,
                        interval: str = "1m") -> pd.DataFrame:
    """Binance K 線（last/mark/premium/index）。startTime 往後分頁，每次最多 1500 根。"""
    if kind not in _BN_KLINE:
        raise ValueError(f"未知 kind: {kind}")
    path, sym_key = _BN_KLINE[kind]
    with_vol = kind == "last"
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {sym_key: symbol, "interval": interval,
                  "startTime": cursor, "endTime": end_ms, "limit": 1500}
        batch = _get(BINANCE, path, params)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + MIN_MS
        if nxt <= cursor or len(batch) < 1500:
            break
        cursor = nxt
        time.sleep(0.4)  # K線 weight≈10，0.4s≈1500 weight/分，留餘裕（0.25 會吃滿 2400）
    if not rows:
        return _empty_kline(with_vol)
    # kline: [openTime,o,h,l,c,volume,closeTime,...]
    df = pd.DataFrame({
        "time": [int(r[0]) for r in rows],
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        **({"volume": [float(r[5]) for r in rows]} if with_vol else {}),
    })
    return _finalize(df, end_ms)


def fetch_bybit_kline(symbol: str, kind: str, start_ms: int, end_ms: int,
                      interval: str = "1") -> pd.DataFrame:
    """Bybit K 線（last/mark/premium/index）。回傳新→舊，用 end 往前分頁，每次最多 1000 根。"""
    if kind not in _BB_KLINE:
        raise ValueError(f"未知 kind: {kind}")
    path = _BB_KLINE[kind]
    with_vol = kind == "last"
    rows = []
    end_cursor = end_ms
    while end_cursor > start_ms:
        params = {"category": "linear", "symbol": symbol, "interval": interval,
                  "start": start_ms, "end": end_cursor, "limit": 1000}
        d = _get(BYBIT, path, params)
        if d.get("retCode") not in (0, None):
            raise RuntimeError(f"Bybit retCode={d.get('retCode')} {d.get('retMsg')}")
        lst = (d.get("result") or {}).get("list") or []
        if not lst:
            break
        rows.extend(lst)
        oldest = int(lst[-1][0])  # list 新→舊，最後一筆最舊
        nxt = oldest - MIN_MS
        if nxt >= end_cursor or len(lst) < 1000:
            break
        end_cursor = nxt
        time.sleep(0.12)  # Bybit 硬限 120 req/s，0.12s≈8 req/s 極安全
    if not rows:
        return _empty_kline(with_vol)
    # list 元素：last=[t,o,h,l,c,vol,turnover]；mark/index/premium=[t,o,h,l,c]
    df = pd.DataFrame({
        "time": [int(r[0]) for r in rows],
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        **({"volume": [float(r[5]) for r in rows]} if with_vol else {}),
    })
    return _finalize(df, end_ms)


def _finalize(df: pd.DataFrame, end_ms: int) -> pd.DataFrame:
    """去重、升序、丟掉未收盤的最後一根（time >= end_ms 視窗外的不要）。"""
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    df = df[df["time"] < end_ms].reset_index(drop=True)
    return df


# ───────────────────────── 資費歷史 ─────────────────────────
def fetch_binance_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Binance 歷史資費。回 time(ms,結算時刻)、rate(float 小數)。limit<=1000，往後分頁。"""
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get(BINANCE, "/fapi/v1/fundingRate",
                     {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        last_t = int(batch[-1]["fundingTime"])
        nxt = last_t + 1
        if nxt <= cursor or len(batch) < 1000:
            break
        cursor = nxt
        time.sleep(0.1)  # fundingRate weight=1，幾乎不占預算
    if not rows:
        return pd.DataFrame(columns=["time", "rate"])
    df = pd.DataFrame({
        "time": [int(r["fundingTime"]) for r in rows],
        "rate": [float(r["fundingRate"]) for r in rows],
    })
    return df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)


def fetch_bybit_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Bybit 歷史資費。回 time(ms)、rate(float)。回傳新→舊，limit<=200，用 end 往前分頁。"""
    rows = []
    end_cursor = end_ms
    while end_cursor > start_ms:
        d = _get(BYBIT, "/v5/market/funding/history",
                 {"category": "linear", "symbol": symbol,
                  "startTime": start_ms, "endTime": end_cursor, "limit": 200})
        if d.get("retCode") not in (0, None):
            raise RuntimeError(f"Bybit retCode={d.get('retCode')} {d.get('retMsg')}")
        lst = (d.get("result") or {}).get("list") or []
        if not lst:
            break
        rows.extend(lst)
        oldest = int(lst[-1]["fundingRateTimestamp"])
        nxt = oldest - 1
        if nxt >= end_cursor or len(lst) < 200:
            break
        end_cursor = nxt
        time.sleep(0.12)  # Bybit 限速寬鬆
    if not rows:
        return pd.DataFrame(columns=["time", "rate"])
    df = pd.DataFrame({
        "time": [int(r["fundingRateTimestamp"]) for r in rows],
        "rate": [float(r["fundingRate"]) for r in rows],
    })
    return df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)


# ───────────────────────── 快取包裝 ─────────────────────────
def _atomic_write(df: pd.DataFrame, fp: Path):
    """原子寫入：先寫 .tmp 再 rename，避免中途中斷留下半截 parquet（時間破洞）。"""
    tmp = fp.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, fp)


def cached_kline(exchange: str, symbol: str, kind: str,
                 start_ms: int, end_ms: int, interval: str) -> pd.DataFrame:
    """K 線抓取的本地 parquet 快取（同窗口同參數命中即不重抓）。
    回測窗口固定，故用 (exchange,symbol,kind,interval,start,end) 當 key。"""
    key = f"{exchange}_{symbol}_{kind}_{interval}_{start_ms}_{end_ms}.parquet"
    fp = CACHE_DIR / key
    if fp.exists():
        return pd.read_parquet(fp)
    if exchange == "binance":
        df = fetch_binance_kline(symbol, kind, start_ms, end_ms, interval)
    elif exchange == "bybit":
        df = fetch_bybit_kline(symbol, kind, start_ms, end_ms, interval)
    else:
        raise ValueError(f"未知 exchange: {exchange}")
    _atomic_write(df, fp)
    return df


def cached_funding(exchange: str, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """歷史資費的本地 parquet 快取（結算資料不可變，極適合快取，避免重跑分析重抓）。"""
    key = f"funding_{exchange}_{symbol}_{start_ms}_{end_ms}.parquet"
    fp = CACHE_DIR / key
    if fp.exists():
        return pd.read_parquet(fp)
    if exchange == "binance":
        df = fetch_binance_funding(symbol, start_ms, end_ms)
    elif exchange == "bybit":
        df = fetch_bybit_funding(symbol, start_ms, end_ms)
    else:
        raise ValueError(f"未知 exchange: {exchange}")
    _atomic_write(df, fp)
    return df


if __name__ == "__main__":
    # 煙霧測試：抓 BTCUSDT 近 3 小時 1m，比對兩家溢價
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    now = int(time.time() * 1000)
    frm = now - 3 * 3600_000
    print("共同合約數:", len(common_symbols()))
    for ex, fn in (("binance", fetch_binance_kline), ("bybit", fetch_bybit_kline)):
        prem = fn("BTCUSDT", "premium", frm, now)
        print(f"{ex} premium 1m: {len(prem)} 根, 最新 close={prem['close'].iloc[-1] if len(prem) else 'NA'}")
