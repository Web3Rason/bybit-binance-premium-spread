# -*- coding: utf-8 -*-
"""
極端溢價時的『價差分布』與『進場價差 → 收斂 EV』分析。

回答兩個問題：
  1. 兩家極端溢價時，價差(|last_spread|)分布長怎樣、能到多大？
  2. 進場價差到多少時，「賭價差收斂」是 +EV？

做法（每事件、無停損無網格，最單純）：
  - 進場：價差首次 ≥ X% 時（延遲 1 分鐘成交，防 look-ahead）
  - 出場：價差收斂到 ≤ EXIT_FLOOR（獲利了結）或持有滿 MAX_HOLD（時限平倉）
  - 毛 PnL = 進場價差 − 出場價差；淨 PnL = 毛 − round-trip 成本
  - 對不同進場門檻 X 統計平均淨 PnL = 該門檻的收斂 EV

用法：
  python convergence_ev.py                  # 預設成本 0.6%(fee0.05+slip0.1)×... round-trip
  python convergence_ev.py --cost 0.3 --cost 0.6 --cost 1.0   # 多成本情境對照
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
EVENTS_DIR = ROOT / "events"

ENTRY_DELAY = 1       # 訊號後延遲分鐘（防 look-ahead）
EXIT_FLOOR = 0.3      # 價差收斂到此 % 即出場
MAX_HOLD_H = 24       # 最長持有
X_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]  # 進場價差門檻 %


def load_events():
    """回 [dict]，每事件含 ts/sp 及 fr/last（算資費用）。"""
    out = []
    for fp in sorted(EVENTS_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if df.empty:
            continue

        def arr(c):
            return (pd.to_numeric(df[c], errors="coerce").to_numpy()
                    if c in df.columns else np.full(len(df), np.nan))

        out.append({
            "symbol": fp.stem.split("_")[0],
            "ts": df["time"].to_numpy("int64"),
            "sp": np.abs(arr("last_spread")) * 100,
            # 自算溢價(分鐘)取兩家較極端者，與圖表/backtest 口徑一致；×100 轉 %
            "prem": np.maximum(np.abs(arr("bn_prem")), np.abs(arr("bb_prem"))) * 100,
            "bn_fr": arr("bn_fr"), "bb_fr": arr("bb_fr"),
            "bn_last": arr("bn_last"), "bb_last": arr("bb_last"),
        })
    return out


def _settle_sum(fr, ei, xi):
    seg = np.nan_to_num(fr[ei:xi + 1])
    if len(seg) < 2:
        return 0.0
    changed = np.diff(seg) != 0
    return float(seg[1:][changed].sum())


def _trade(ev, sig_vals, thr):
    """訊號(sig_vals)首次≥thr 時進場(延遲1m)；進場價與出場一律看『價差』，
    收斂到 EXIT_FLOOR 或 24h 出場。sig_vals 可為價差(ev['sp'])或溢價(ev['prem'])。
    回 (毛PnL含資費, 收斂?, MAE, 持有分, 資費%)。"""
    ts, sp = ev["ts"], ev["sp"]
    sig = (~np.isnan(sig_vals)) & (sig_vals >= thr)
    if not sig.any():
        return None
    ai = int(np.argmax(sig)) + ENTRY_DELAY
    if ai >= len(sp) or np.isnan(sp[ai]):
        return None
    S_in, t_in = sp[ai], ts[ai]
    post_sp, post_ts = sp[ai + 1:], ts[ai + 1:]
    if len(post_sp) == 0:
        return None
    v = ~np.isnan(post_sp)
    conv = v & (post_sp <= EXIT_FLOOR)
    tout = v & ((post_ts - t_in) >= MAX_HOLD_H * 3_600_000)
    if conv.any():
        k = int(np.argmax(conv)); converged = True
    elif tout.any():
        k = int(np.argmax(tout)); converged = False
    else:
        idx = np.where(v)[0]
        if len(idx) == 0:
            return None
        k = int(idx[-1]); converged = False
    # MAE：進場後到出場前，價差最大擴大到多少（=要扛的浮虧，決定保證金/爆倉風險）
    seg = post_sp[:k + 1]
    mae = float(np.nanmax(seg)) - S_in if len(seg) else 0.0
    hold_min = (post_ts[k] - t_in) / 60000

    # 資費：short 高價家 perp 每次結算收 +fr_H、long 低價家付 −fr_L（小數×100=%）
    xi = ai + 1 + k
    bn_in, bb_in = ev["bn_last"][ai], ev["bb_last"][ai]
    if not (bn_in == bn_in) or not (bb_in == bb_in):
        funding = 0.0
    else:
        fr_H, fr_L = (ev["bn_fr"], ev["bb_fr"]) if bn_in >= bb_in else (ev["bb_fr"], ev["bn_fr"])
        funding = (_settle_sum(fr_H, ai, xi) - _settle_sum(fr_L, ai, xi)) * 100

    gross = (S_in - post_sp[k]) + funding   # 價差收斂 + 資費
    return gross, converged, max(0.0, mae), hold_min, funding


def trade_at(ev, X):
    """價差首次≥X% 進場，賭收斂。"""
    return _trade(ev, ev["sp"], X)


def trade_at_premium(ev, P):
    """溢價首次衝到≥P%（=告警觸發那一刻）進場，賭價差收斂。"""
    return _trade(ev, ev["prem"], P)


PREM_GRID = [2.0, 5.0, 8.0, 12.0, 20.0, 40.0]   # 溢價門檻 %（對齊告警）
COST_REF = 0.6   # 淨EV 參考成本（round-trip %）


def _agg_rows(data, grid, trade_fn):
    """對門檻網格逐一回測，回 [dict] 供 print 與 JSON 共用。"""
    rows = []
    for thr in grid:
        trades = [t for t in (trade_fn(ev, thr) for ev in data) if t]
        if not trades:
            continue
        gross = np.array([t[0] for t in trades])
        mae = np.array([t[2] for t in trades])
        rows.append({
            "thr": thr,
            "n": len(trades),
            "conv_rate": round(float(np.mean([t[1] for t in trades])), 4),
            "gross_ev": round(float(gross.mean()), 3),
            "funding": round(float(np.mean([t[4] for t in trades])), 4),
            "net_ev": round(float(gross.mean() - COST_REF), 3),
            "mae_mean": round(float(mae.mean()), 3),
            "mae_p90": round(float(np.percentile(mae, 90)), 3),
            "hold_h_median": round(float(np.median([t[3] for t in trades]) / 60), 2),
            "ev_mae": round(float(gross.mean() / mae.mean()), 3) if mae.mean() > 0 else None,
        })
    return rows


def _print_rows(title, axis_label, rows):
    print(f"\n=== {title}（出場：收斂到≤{EXIT_FLOOR}% 或 {MAX_HOLD_H}h）===")
    print(f"{axis_label}  觸發  收斂率  毛EV%  其中資費%  淨EV@{COST_REF}%  均MAE%  P90_MAE%  中位持有  EV/MAE")
    for r in rows:
        print(f"  ≥{r['thr']:<4g} {r['n']:5d}  {r['conv_rate']*100:4.0f}%  {r['gross_ev']:+6.2f}  "
              f"{r['funding']:+7.3f}  {r['net_ev']:+7.2f}  {r['mae_mean']:6.2f}  {r['mae_p90']:7.2f}  "
              f"{r['hold_h_median']:6.1f}h  {r['ev_mae'] if r['ev_mae'] is not None else '—'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, action="append",
                    help="round-trip 成本 %（可多次，對照不同滑點情境）")
    args = ap.parse_args()

    data = load_events()
    print(f"[收斂EV] 載入 {len(data)} 個極端溢價事件（已含資費計算）\n")

    # ── 1. 價差分布（每事件的最大 |價差|）──
    peaks = np.array([np.nanmax(ev["sp"]) for ev in data if np.any(~np.isnan(ev["sp"]))])
    dist_buckets = []
    for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 1e9)]:
        c = int(((peaks >= lo) & (peaks < hi)).sum())
        dist_buckets.append({"label": (f"≥{lo:g}%" if hi > 1e8 else f"{lo:g}~{hi:g}%"),
                             "n": c, "pct": round(c / len(peaks) * 100, 1)})
    print("=== 極端溢價時的價差分布（每事件最大 |last_spread|）===")
    print(f"  中位數 {np.median(peaks):.2f}%　平均 {peaks.mean():.2f}%")
    for q in (50, 75, 90, 95, 99):
        print(f"  P{q}: {np.percentile(peaks, q):.2f}%", end="  ")
    print()
    for b in dist_buckets:
        print(f"  價差 {b['label']}: {b['n']} 事件 ({b['pct']:.0f}%)")

    # ── 2. 兩種進場訊號 → 收斂 EV ──
    by_spread = _agg_rows(data, X_GRID, trade_at)
    by_premium = _agg_rows(data, PREM_GRID, trade_at_premium)
    _print_rows("溢價門檻 → 賭收斂（溢價衝到 P% 進場）", "溢價P ", by_premium)
    _print_rows("進場價差門檻 → 賭收斂", "進場X ", by_spread)

    print(f"\n關鍵解讀：")
    print(f"  ・收斂率~100% → 極端溢價『最終幾乎必收斂』，但這假設你能扛到收斂。")
    print(f"  ・真正風險是 MAE：進場後價差平均/P90 還會擴大多少 = 要準備的保證金、被掃爆風險。")
    print(f"  ・EV/MAE >1 才代表期望收益大於要扛的浮虧；資費(FR)在持有期多為小幅淨支出，非收益來源。")

    # ── 3. 輸出 JSON 供前端讀取 ──
    out = {
        "n_events": len(data),
        "params": {"exit_floor": EXIT_FLOOR, "max_hold_h": MAX_HOLD_H,
                   "entry_delay_min": ENTRY_DELAY, "cost_ref": COST_REF},
        "spread_dist": {
            "median": round(float(np.median(peaks)), 2), "mean": round(float(peaks.mean()), 2),
            "pcts": {f"p{q}": round(float(np.percentile(peaks, q)), 2) for q in (50, 75, 90, 95, 99)},
            "buckets": dist_buckets,
        },
        "by_premium_thr": by_premium,
        "by_entry_spread": by_spread,
    }
    OUT_JSON = ROOT / "convergence_ev.json"
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[收斂EV] 已輸出 → {OUT_JSON.name}")


if __name__ == "__main__":
    main()
