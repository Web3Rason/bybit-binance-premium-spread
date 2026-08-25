# -*- coding: utf-8 -*-
"""
階段 2：事件深抓 —— 對候選幣的事件窗抓兩家 1m 資料並對齊，算各種 spread。

口徑（重要，實測 2026-06-16）：
  兩家『官方 premium index K 線』口徑不同（Binance 保留盤中瞬時尖峰可達數十%，
  Bybit 經 impact/平滑後僅 ±0.3%），不可直接相減。故『溢價』一律自己用
  (mark − index) / index 重算，兩家同口徑才可比。官方 premium 僅留作對照欄。

每分鐘對齊後計算：
  bn_prem / bb_prem   = (mark − index)/index            ← 統一口徑瞬時溢價
  prem_spread         = bb_prem − bn_prem               ← 溢價背離（驅動源）
  fr_spread           = bb_fr − bn_fr                    ← 資費背離（驗證核心假設）
  mark_spread         = (bb_mark − bn_mark)/bn_mark      ← 標記價差（套利結果）
  last_spread         = (bb_last − bn_last)/bn_last      ← 真實成交價差
  index_spread        = (bb_index − bn_index)/bn_index   ← 指數差（排除取樣干擾用）

守機器負載規則：單程序序列，逐事件窗處理，每窗 1m 資料量小（幾天=幾千根）。
用 --top-symbols / --max-events 控管總量，預設保守。

用法：
  python fetch_event.py                       # 用 candidates.json，預設前 30 幣
  python fetch_event.py --top-symbols 50 --max-events 8
  python fetch_event.py --only TRBUSDT        # 只處理指定幣
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

import data

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CAND = ROOT / "candidates.json"
EVENTS_DIR = ROOT / "events"
EVENTS_DIR.mkdir(exist_ok=True)
INDEX_JSON = ROOT / "events_index.json"


# 抓 funding 時往前多抓一個最大結算週期，確保窗起點 frm 之前有一筆已結算費率，
# 讓 merge_asof(backward) 在第一根 K 線就填得到值（否則窗前段 fr/fr_usage 全 NaN）。
# 也讓 infer_interval_h 多幾個結算點、週期推算更穩。
FUNDING_LOOKBACK_MS = 8 * 3_600_000


def _kline(ex: str, symbol: str, kind: str, frm: int, to: int) -> pd.DataFrame:
    """抓 1m K 線。mark/index/premium 只需 close（算溢價/價差）；
    last 額外保留 OHLC（前端畫 K 線用），欄名加 pfx 前綴於呼叫端處理。"""
    iv = "1m" if ex == "binance" else "1"
    try:
        df = data.cached_kline(ex, symbol, kind, frm, to, iv)
    except Exception as e:
        print(f"    [warn] {ex} {symbol} {kind} 失敗: {e}")
        return pd.DataFrame()
    if df.empty:
        return df
    if kind == "last":
        return df[["time", "open", "high", "low", "close"]]
    return df[["time", "close"]]


def _funding(ex: str, symbol: str, frm: int, to: int) -> pd.DataFrame:
    try:
        return data.cached_funding(ex, symbol, frm - FUNDING_LOOKBACK_MS, to)
    except Exception as e:
        print(f"    [warn] {ex} {symbol} funding 失敗: {e}")
        return pd.DataFrame(columns=["time", "rate"])


def _usage(rate, cap_pos, cap_neg):
    """資費使用率（貼頂程度，0~1+）：正費率除以上限、負費率除以下限。"""
    import numpy as np
    r = pd.to_numeric(rate, errors="coerce")
    up = cap_pos if (cap_pos is not None and cap_pos != 0) else np.nan
    lo = cap_neg if (cap_neg is not None and cap_neg != 0) else np.nan
    return np.where(r >= 0, r / up, r / lo)


def build_event(symbol: str, frm: int, to: int,
                bn_info: dict, bb_info: dict | None) -> tuple[pd.DataFrame, dict] | None:
    """抓兩家 1m mark/index/last/premium + funding，對齊算各 spread。
    bn_info: {cap, floor, interval_h}（Binance 當前設定）
    bb_info: {interval_min, upper, lower}（Bybit 當前設定）或 None
    回 (對齊後 df, 事件級 meta)。"""
    cols = {}
    for ex, pfx in (("binance", "bn"), ("bybit", "bb")):
        for kind in ("mark", "index", "last", "premium"):
            df = _kline(ex, symbol, kind, frm, to)
            if df.empty:
                continue
            if kind == "last":
                # 保留 OHLC：close→{pfx}_last（算價差），OHLC→{pfx}_last_o/h/l/c（畫 K 線）
                cols[f"{pfx}_last"] = df.rename(columns={
                    "open": f"{pfx}_last_o", "high": f"{pfx}_last_h",
                    "low": f"{pfx}_last_l", "close": f"{pfx}_last"})
            else:
                cols[f"{pfx}_{kind}"] = df.rename(columns={"close": f"{pfx}_{kind}"})

    need = ["bn_mark", "bn_index", "bb_mark", "bb_index"]
    if any(k not in cols for k in need):
        return None  # 缺核心欄位無法算溢價/價差

    # 以 mark 的時間軸為基準，inner join 對齊（兩家整分鐘戳一致）
    m = cols["bn_mark"]
    for k, df in cols.items():
        if k == "bn_mark":
            continue
        m = m.merge(df, on="time", how="inner")
    if m.empty:
        return None

    # funding：往前填充到每分鐘（最近一次結算費率代表當時資費水平）
    # 同時記錄事件期間『實際結算週期』（從結算時間戳推算，反映極端期動態縮短）
    meta = {}
    for ex, pfx in (("binance", "bn"), ("bybit", "bb")):
        fr = _funding(ex, symbol, frm, to)
        if fr.empty:
            m[f"{pfx}_fr"] = pd.NA
            meta[f"{pfx}_actual_interval_h"] = None
        else:
            meta[f"{pfx}_actual_interval_h"] = data.infer_interval_h(fr["time"].tolist())
            fr = fr.rename(columns={"rate": f"{pfx}_fr"}).sort_values("time")
            m = pd.merge_asof(m.sort_values("time"), fr, on="time", direction="backward")

    # 兩家 cap / 當前週期（解釋層；歷史 cap 不可得，用當前值近似並於報告註明）
    bn_cap, bn_floor = bn_info["cap"], bn_info["floor"]
    _bu = (bb_info or {}).get("upper")
    _bl = (bb_info or {}).get("lower")
    bb_up = _bu if _bu is not None else data.BINANCE_DEFAULT_CAP
    bb_low = _bl if _bl is not None else -data.BINANCE_DEFAULT_CAP
    meta.update({
        "bn_cap": bn_cap, "bn_floor": bn_floor, "bn_interval_h": bn_info["interval_h"],
        "bb_cap": bb_up, "bb_floor": bb_low,
        "bb_interval_h": (bb_info or {}).get("interval_min", 480) / 60,
    })
    m["bn_fr_usage"] = _usage(m.get("bn_fr"), bn_cap, bn_floor)
    m["bb_fr_usage"] = _usage(m.get("bb_fr"), bb_up, bb_low)

    # 統一口徑溢價
    m["bn_prem"] = (m["bn_mark"] - m["bn_index"]) / m["bn_index"]
    m["bb_prem"] = (m["bb_mark"] - m["bb_index"]) / m["bb_index"]
    m["prem_spread"] = m["bb_prem"] - m["bn_prem"]
    m["mark_spread"] = (m["bb_mark"] - m["bn_mark"]) / m["bn_mark"]
    m["last_spread"] = ((m["bb_last"] - m["bn_last"]) / m["bn_last"]
                        if "bb_last" in m and "bn_last" in m else pd.NA)
    m["index_spread"] = (m["bb_index"] - m["bn_index"]) / m["bn_index"]
    if "bn_fr" in m and "bb_fr" in m:
        m["fr_spread"] = m["bb_fr"] - m["bn_fr"]

    m = m.sort_values("time").reset_index(drop=True)
    meta["bn_fr_usage_max"] = round(float(pd.to_numeric(m["bn_fr_usage"]).abs().max()), 3)
    meta["bb_fr_usage_max"] = round(float(pd.to_numeric(m["bb_fr_usage"]).abs().max()), 3)
    return m, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-symbols", type=int, default=30, help="處理峰值最大的前 N 個候選幣")
    ap.add_argument("--max-events", type=int, default=5, help="每幣最多深抓的事件窗數（取峰值最大者）")
    ap.add_argument("--only", type=str, default="", help="只處理指定幣（逗號分隔）")
    ap.add_argument("--max-window-days", type=int, default=14,
                    help="單事件窗 1m 深抓的最大天數（防呆，超過裁到峰值日附近）")
    args = ap.parse_args()

    if not CAND.exists():
        sys.exit("找不到 candidates.json，請先跑 scan.py")
    cand = json.loads(CAND.read_text(encoding="utf-8"))
    candidates = cand["candidates"]

    if args.only:
        only = {s.strip().upper() for s in args.only.split(",")}
        candidates = [c for c in candidates if c["symbol"] in only]
    else:
        candidates = candidates[:args.top_symbols]

    # 一次性載入兩家 cap/週期 metadata（解釋層）
    print("[fetch_event] 載入兩家資費 cap/週期設定 ...")
    bn_fi = data.binance_funding_info()
    bb_inst = data.bybit_instruments()
    bn_default = {"cap": data.BINANCE_DEFAULT_CAP, "floor": -data.BINANCE_DEFAULT_CAP,
                  "interval_h": data.BINANCE_DEFAULT_INTERVAL_H}

    index = []
    for ci, c in enumerate(candidates, 1):
        sym = c["symbol"]
        events = sorted(c["events"], key=lambda e: abs(e["peak_premium_pct"]), reverse=True)
        events = events[:args.max_events]
        for ei, ev in enumerate(events):
            frm, to = ev["start_ms"], ev["end_ms"]
            # 防呆：單窗過大會撐爆 1m 抓取（機器負載），裁到峰值日附近
            max_ms = args.max_window_days * 86_400_000
            if to - frm > max_ms:
                pk = ev.get("peak_day_ms", (frm + to) // 2)
                frm, to = pk - max_ms // 2, pk + max_ms // 2
                print(f"    [防呆] 窗過大，裁切為峰值日 ±{args.max_window_days//2} 天")
            print(f"({ci}/{len(candidates)}) {sym} 事件{ei+1} "
                  f"峰值{ev['peak_premium_pct']:.2f}%@{ev['peak_ex']} ...")
            built = build_event(sym, frm, to, bn_fi.get(sym, bn_default), bb_inst.get(sym))
            if built is None or built[0].empty:
                print("    略過（資料不足）")
                continue
            df, meta = built
            fp = EVENTS_DIR / f"{sym}_{frm}.parquet"
            df.to_parquet(fp, index=False)
            index.append({
                "symbol": sym, "file": fp.name,
                "start_ms": frm, "end_ms": to,
                "peak_premium_pct": ev["peak_premium_pct"], "peak_ex": ev["peak_ex"],
                "rows": len(df),
                "max_abs_mark_spread_pct": round(float(df["mark_spread"].abs().max()) * 100, 4),
                "max_abs_prem_spread_pct": round(float(df["prem_spread"].abs().max()) * 100, 4),
                "max_abs_last_spread_pct": round(float(pd.to_numeric(df["last_spread"]).abs().max()) * 100, 4),
                **meta,
            })
            print(f"    存 {len(df)} 列 → {fp.name}  "
                  f"最大|mark_spread|={index[-1]['max_abs_mark_spread_pct']:.3f}%  "
                  f"週期 BN={meta.get('bn_actual_interval_h')}h/BB={meta.get('bb_actual_interval_h')}h  "
                  f"使用率max BN={meta['bn_fr_usage_max']}/BB={meta['bb_fr_usage_max']}")

    index.sort(key=lambda x: x["max_abs_mark_spread_pct"], reverse=True)
    INDEX_JSON.write_text(json.dumps(
        {"generated_at": int(time.time() * 1000), "n_events": len(index), "events": index},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[fetch_event] 完成：{len(index)} 個事件窗 → events/ + {INDEX_JSON.name}")


if __name__ == "__main__":
    main()
