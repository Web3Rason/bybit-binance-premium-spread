# -*- coding: utf-8 -*-
"""
階段 1：全市場粗篩 —— 找出「曾出現極端溢價」的幣與事件時間窗。

策略（守機器負載規則：單程序序列、低粒度、資料量小）：
  對每個『兩家共同上市』的 USDT 永續，抓兩家的『daily premium index K 線』
  （daily 一次 API 即可包 2~4 年），看每日 max(|high|,|low|) 是否觸及極端閾值。
  觸發日合併成事件窗（相鄰日合併 + 前後 pad），輸出 candidates.json，
  供階段 2 (fetch_event.py) 只對候選窗深抓 1m。

  不做成交額/流動性篩選 —— 只要溢價有極端就納入（依使用者指示）。

用法：
  python scan.py                  # 全市場，預設回溯盡量長
  python scan.py --limit 10       # 只掃前 10 個（測試）
  python scan.py --threshold 0.005 --days 1460
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import data

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent / "candidates.json"
DAY_MS = 86_400_000

# daily K 線的 interval 字串（兩家不同）
_DAILY = {"binance": "1d", "bybit": "D"}


def _scan_symbol(symbol: str, start_ms: int, end_ms: int, threshold: float) -> list[dict]:
    """回該幣的觸發日清單：[{day_ms, ex, premium_pct}]（兩家分開記）。"""
    hits = []
    for ex in ("binance", "bybit"):
        fn = data.fetch_binance_kline if ex == "binance" else data.fetch_bybit_kline
        try:
            df = fn(symbol, "premium", start_ms, end_ms, _DAILY[ex])
        except Exception as e:
            print(f"  [warn] {ex} {symbol} 抓取失敗: {e}")
            continue
        time.sleep(0.1)  # symbol/交易所間隔（daily 無分頁，不靠分頁內 sleep 控速）
        if df.empty:
            continue
        # 當日最極端溢價 = 絕對值較大的那一側(high 或 low)；向量化過濾觸發日
        hi_abs, lo_abs = df["high"].abs(), df["low"].abs()
        peak = np.maximum(hi_abs, lo_abs)
        signed = np.where(hi_abs >= lo_abs, df["high"], df["low"])
        mask = (peak >= threshold).to_numpy()
        for t, s in zip(df["time"].to_numpy()[mask], signed[mask]):
            hits.append({"day_ms": int(t), "ex": ex,
                         "premium_pct": round(float(s) * 100, 4)})
    return hits


def _merge_events(hits: list[dict], gap_days: int, window_days: int,
                  start_ms: int, end_ms: int) -> list[dict]:
    """把觸發日依時間分段成『連續極端期』，每段以峰值日為中心取固定窗。

    為何不用整段當窗：長期持續極端的幣（如某些幣天天貼溢價）會合併出跨數月的
    巨窗，1m 深抓會撐爆記憶體/流量（違反機器負載規則）。研究目的是看『極端溢價
    當下價差怎麼走』，故聚焦峰值日 ± window_days/2 的固定窗即可。"""
    if not hits:
        return []
    days = sorted({h["day_ms"] for h in hits})
    segments = []
    cur_start = cur_end = days[0]
    for d in days[1:]:
        if d - cur_end <= gap_days * DAY_MS:
            cur_end = d
        else:
            segments.append((cur_start, cur_end))
            cur_start = cur_end = d
    segments.append((cur_start, cur_end))

    half = window_days * DAY_MS // 2
    events = []
    for ss, se in segments:
        in_seg = [h for h in hits if ss <= h["day_ms"] <= se]
        peak = max(in_seg, key=lambda h: abs(h["premium_pct"]))
        pk = peak["day_ms"]
        events.append({
            "start_ms": max(start_ms, pk - half),
            "end_ms": min(end_ms, pk + half + DAY_MS),
            "peak_premium_pct": peak["premium_pct"],
            "peak_ex": peak["ex"],
            "peak_day_ms": pk,
            "segment_days": round((se - ss) / DAY_MS) + 1,  # 連續極端期長度（資訊用）
            "trigger_days": len(set(h["day_ms"] for h in in_seg)),
        })
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只掃前 N 個共同合約（測試）")
    ap.add_argument("--days", type=int, default=1460, help="回溯天數（預設 4 年，API 自然截斷）")
    ap.add_argument("--threshold", type=float, default=0.005,
                    help="極端溢價閾值（小數，0.005=0.5%%），任一家當日 |premium| 觸及即納入")
    ap.add_argument("--gap-days", type=int, default=2, help="相鄰觸發日合併為同一連續極端期的間隔")
    ap.add_argument("--window-days", type=int, default=7,
                    help="事件窗總天數（以峰值日為中心，控制 1m 深抓量）")
    args = ap.parse_args()

    now = int(time.time() * 1000)
    end_ms = (now // DAY_MS) * DAY_MS          # 今天 0 點 UTC（不含未收盤當日）
    start_ms = end_ms - args.days * DAY_MS

    syms = data.common_symbols()
    if args.limit:
        syms = syms[:args.limit]
    print(f"[scan] 共同合約 {len(syms)} 個，閾值 |premium|>={args.threshold*100:.3f}%，"
          f"回溯 {args.days} 天")

    candidates = []
    for i, sym in enumerate(syms, 1):
        hits = _scan_symbol(sym, start_ms, end_ms, args.threshold)
        events = _merge_events(hits, args.gap_days, args.window_days, start_ms, end_ms)
        if events:
            top = max(events, key=lambda e: abs(e["peak_premium_pct"]))
            candidates.append({"symbol": sym, "events": events,
                               "max_premium_pct": top["peak_premium_pct"]})
            print(f"  ({i}/{len(syms)}) {sym}: {len(events)} 事件窗, "
                  f"峰值 {top['peak_premium_pct']:.3f}% @ {top['peak_ex']}")
        else:
            print(f"  ({i}/{len(syms)}) {sym}: 無極端事件", end="\r")

    candidates.sort(key=lambda c: abs(c["max_premium_pct"]), reverse=True)
    out = {
        "generated_at": now,
        "params": {"days": args.days, "threshold": args.threshold,
                   "gap_days": args.gap_days, "window_days": args.window_days,
                   "start_ms": start_ms, "end_ms": end_ms},
        "n_scanned": len(syms),
        "n_candidates": len(candidates),
        "candidates": candidates,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[scan] 完成：{len(candidates)}/{len(syms)} 個幣有極端溢價事件 → {OUT.name}")


if __name__ == "__main__":
    main()
