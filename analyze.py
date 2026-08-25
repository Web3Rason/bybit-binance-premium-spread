# -*- coding: utf-8 -*-
"""
階段 3：分析 —— 主軸『兩家溢價 → 兩家價差』。

輸出改為『互動圖資料』(events_json/*.json)，由前端 index.html 用 TradingView
Lightweight Charts 畫互動式多窗格時序圖（可縮放/平移/hover）。不再產靜態 PNG。

研究主關係：
  溢價端  bn_prem / bb_prem = (mark−index)/index（統一口徑）、prem_spread = bb_prem−bn_prem
  價差端  last_spread = (bb_last−bn_last)/bn_last  ← 真實成交價差（主要結果軸）
  統計用 Spearman 秩相關（抗單分鐘插針）、P95/P99（穩健幅度，取代會被插針主導的 max）。
  以指數差中位數剔除同名不同幣假事件。

每事件輸出：events_json/{name}.json（時序）；彙總統計 analysis.json。

用法：python analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
EVENTS_DIR = ROOT / "events"
SERIES_DIR = ROOT / "events_json"
SERIES_DIR.mkdir(exist_ok=True)
INDEX_JSON = ROOT / "events_index.json"
ANALYSIS_JSON = ROOT / "analysis.json"

# 指數差中位數超過此值(%)→判定同名不同幣，價差比較無意義
SUSPECT_INDEX_PCT = 2.0


def _safe(v) -> float | None:
    """擋 NaN/inf，避免寫進 JSON 變非法（前端 JSON.parse 會掛）。"""
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    return None if (f != f or f in (float("inf"), float("-inf"))) else round(f, 4)


def _corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float | None:
    """相關係數。method='spearman' 為秩相關，對單點插針離群天生穩健（主結論用）。"""
    if a is None or b is None:
        return None
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 10 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return None
    return _safe(d.iloc[:, 0].corr(d.iloc[:, 1], method=method))


def _pct_quantile(s: pd.Series, q: float) -> float | None:
    """|序列| 的分位數(%)，對插針穩健（取代會被單分鐘尖峰主導的 max）。"""
    if s is None:
        return None
    v = pd.to_numeric(s, errors="coerce").abs().quantile(q)
    return _safe(v * 100)


def _lead_lag(driver: pd.Series, result: pd.Series, max_lag: int = 30) -> dict | None:
    """driver 領先 result 幾分鐘時相關最強（正 lag = 溢價領先價差）。"""
    d = pd.concat([driver.rename("d"), result.rename("r")], axis=1).dropna().reset_index(drop=True)
    if len(d) < 3 * max_lag:
        return None
    best_lag, best_r = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = d["d"].iloc[:len(d) - lag], d["r"].iloc[lag:]
        else:
            x, y = d["d"].iloc[-lag:], d["r"].iloc[:len(d) + lag]
        if x.std() == 0 or y.std() == 0:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if abs(r) > abs(best_r):
            best_r, best_lag = float(r), lag
    return {"lead_lag_min": best_lag, "corr_at_best": round(best_r, 4)}


def _dump_timeseries(df: pd.DataFrame, meta: dict) -> str:
    """輸出該事件的時序資料 JSON（columnar），供前端 Lightweight Charts 畫圖。
    time 轉 unix 秒；溢價/價差轉 %；NaN→null（前端跳過該點）。"""
    def col(name, mul=1.0):
        if name not in df:
            return None
        x = pd.to_numeric(df[name], errors="coerce") * mul
        return [None if (v != v) else round(float(v), 6) for v in x]

    data = {
        "symbol": meta["symbol"],
        "peak_premium_pct": meta.get("peak_premium_pct"),
        "peak_ex": meta.get("peak_ex"),
        "time": [int(t // 1000) for t in df["time"]],
        "bn_prem": col("bn_prem", 100), "bb_prem": col("bb_prem", 100),
        "prem_spread": col("prem_spread", 100), "last_spread": col("last_spread", 100),
        "mark_spread": col("mark_spread", 100),
        "bn_fr_usage": col("bn_fr_usage"), "bb_fr_usage": col("bb_fr_usage"),
        # 原始資費率(%)：供「七夜：溢價↔資費穿插」窗格疊圖（bb_fr 已逐分鐘 ffill）
        "bn_fr": col("bn_fr", 100), "bb_fr": col("bb_fr", 100),
        # 兩家 last price 的 OHLC（前端畫 K 線；無對應欄則為 None）
        "bn_last_o": col("bn_last_o"), "bn_last_h": col("bn_last_h"),
        "bn_last_l": col("bn_last_l"), "bn_last_c": col("bn_last"),
        "bb_last_o": col("bb_last_o"), "bb_last_h": col("bb_last_h"),
        "bb_last_l": col("bb_last_l"), "bb_last_c": col("bb_last"),
    }
    name = meta["file"].replace(".parquet", ".json")
    (SERIES_DIR / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return name


def analyze_event(meta: dict) -> dict | None:
    fp = EVENTS_DIR / meta["file"]
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if df.empty:
        return None
    for c in ("bn_prem", "bb_prem", "prem_spread", "last_spread", "mark_spread",
              "index_spread", "bn_fr_usage", "bb_fr_usage"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    stat = {
        "symbol": meta["symbol"], "file": meta["file"], "series": None,
        "peak_premium_pct": meta.get("peak_premium_pct"),
        "peak_ex": meta.get("peak_ex"),
        "start_ms": meta.get("start_ms"),  # 事件起始時間(供前端顯示日期，區分最近/歷史)
        "rows": len(df),
        # 主關係：溢價差 → 真實成交價差。Spearman 為主（抗插針），Pearson 對照
        "spearman_premSpread_lastSpread": _corr(df.get("prem_spread"), df.get("last_spread"), "spearman"),
        "corr_premSpread_lastSpread": _corr(df.get("prem_spread"), df.get("last_spread")),
        "leadlag_premSpread_to_lastSpread": _lead_lag(df["prem_spread"], df["last_spread"])
            if "prem_spread" in df and "last_spread" in df else None,
        # 價差幅度用穩健分位數(P95/P99)，不用會被單分鐘插針主導的 max
        "p99_abs_last_spread_pct": _pct_quantile(df.get("last_spread"), 0.99),
        "p95_abs_last_spread_pct": _pct_quantile(df.get("last_spread"), 0.95),
        "max_abs_last_spread_pct": _pct_quantile(df.get("last_spread"), 1.0),
        "max_abs_prem_spread_pct": _pct_quantile(df.get("prem_spread"), 1.0),
        "max_abs_index_spread_pct": _pct_quantile(df.get("index_spread"), 1.0),
        "median_abs_index_spread_pct": _pct_quantile(df.get("index_spread"), 0.5),
        # 解釋層
        "bn_actual_interval_h": meta.get("bn_actual_interval_h"),
        "bb_actual_interval_h": meta.get("bb_actual_interval_h"),
        "bn_cap": meta.get("bn_cap"), "bb_cap": meta.get("bb_cap"),
        "bn_fr_usage_max": meta.get("bn_fr_usage_max"),
        "bb_fr_usage_max": meta.get("bb_fr_usage_max"),
    }
    mid = stat["median_abs_index_spread_pct"]
    stat["suspect_diff_coin"] = bool(mid is not None and mid > SUSPECT_INDEX_PCT)
    stat["series"] = _dump_timeseries(df, meta)
    return stat


def main():
    if not INDEX_JSON.exists():
        sys.exit("找不到 events_index.json，請先跑 fetch_event.py")
    idx = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    results = []
    for meta in idx["events"]:
        print(f"分析 {meta['symbol']} {meta['file']} ...")
        st = analyze_event(meta)
        if st:
            results.append(st)
            print(f"  spearman={st['spearman_premSpread_lastSpread']}  "
                  f"P99|last|={st['p99_abs_last_spread_pct']}%  → {st['series']}")
    results.sort(key=lambda x: (x.get("p99_abs_last_spread_pct") or 0), reverse=True)
    clean = [r for r in results if not r["suspect_diff_coin"]]
    suspect = [r for r in results if r["suspect_diff_coin"]]
    sp = [r["spearman_premSpread_lastSpread"] for r in clean
          if r.get("spearman_premSpread_lastSpread") is not None]
    summary = {
        "n_events": len(results),
        "n_clean": len(clean), "n_suspect": len(suspect),
        "clean_spearman_median": round(float(np.median(sp)), 4) if sp else None,
        "suspect_index_pct_threshold": SUSPECT_INDEX_PCT,
    }
    ANALYSIS_JSON.write_text(json.dumps(
        {**summary, "events": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[analyze] 完成 {len(results)} 事件（可信 {len(clean)} / 疑似不同幣 {len(suspect)}）"
          f"  可信 Spearman 中位數={summary['clean_spearman_median']}  → events_json/ + {ANALYSIS_JSON.name}")


if __name__ == "__main__":
    main()
