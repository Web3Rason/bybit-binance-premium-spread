# -*- coding: utf-8 -*-
"""
策略回測 v2：極端溢價 → 價差收斂套利（delta-neutral pair trade）。

相對 v1 的務實化（經第二輪審查後調整）：
  1. 進場延遲：訊號出現後延遲 ENTRY_DELAY 分鐘才成交（防 look-ahead —— 極端溢價
     只存在數秒，用訊號當下價進場是作弊）。
  2. 滑點：每腿成本 = taker_fee + slippage；pair 進出共 4 腿。
  3. 擴網格：參數往邊界外擴，檢驗最佳組是否為「角落解」（過擬合）。
  4. 樣本外：事件按時間 70/30 切分，in-sample 選參、out-sample 只驗證（不調參）。
  5. 穩健選參：選「參數鄰域(±1格)平均報酬」最高者 → 高原而非尖峰。
  6. 風險指標：逐筆 PnL → 最大回撤、盈虧比、肥尾檢定（移除 top5% 贏家後是否仍正）。

PnL(%)：(進場價差 − 出場價差) − 4×(fee+slip) + 持有期資費淨額（資費為小數×100）。

用法：
  python backtest.py                 # 預設 fee 0.05%、slip 0.10%/腿
  python backtest.py --slip 0.2      # 壓力測試：冷門幣較大滑點
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
EVENTS_DIR = ROOT / "events"
OUT = ROOT / "backtest_result.json"

ENTRY_DELAY = 1   # 訊號後延遲幾分鐘成交（look-ahead 防護）
OOS_FRAC = 0.30   # 後 30% 事件當樣本外

# 擴張後的參數網格（往邊界外擴，破角落解）
GRID = {
    "premium_thr": [2.0, 5.0, 8.0, 12.0],   # 極端溢價訊號門檻 %
    "entry_sp":    [1.0, 2.0, 3.0],          # 進場價差門檻 %
    "exit_sp":     [0.2, 0.5],               # 收斂出場目標 %
    "stop_sp":     [3.0, 5.0, 8.0, 12.0],    # 停損(價差續擴) %
    "max_hold_h":  [1, 4, 24],               # 最長持有(小時)
}
GKEYS = list(GRID.keys())


def load_events() -> list[dict]:
    evs = []
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

        ts = df["time"].to_numpy(dtype="int64")
        evs.append({
            "symbol": fp.stem.split("_")[0],
            "start_ms": int(fp.stem.split("_")[-1]),   # 事件起始時間（供時間切分）
            "ts": ts,
            "sp": np.abs(arr("last_spread")) * 100,
            "prem": np.maximum(np.abs(arr("bn_prem")), np.abs(arr("bb_prem"))) * 100,
            "bn_fr": arr("bn_fr"), "bb_fr": arr("bb_fr"),
            "bn_last": arr("bn_last"), "bb_last": arr("bb_last"),
        })
    return evs


def _settle_sum(fr, ei, xi):
    seg = np.nan_to_num(fr[ei:xi + 1])
    if len(seg) < 2:
        return 0.0
    changed = np.diff(seg) != 0
    return float(seg[1:][changed].sum())


def sim_event(ev, p, cost4):
    """單事件單組參數，回交易結果或 None。cost4 = 4×(fee+slip)（已 %）。"""
    ts, sp, prem = ev["ts"], ev["sp"], ev["prem"]
    valid = ~np.isnan(sp)
    enter = valid & (prem >= p["premium_thr"]) & (sp >= p["entry_sp"])
    if not enter.any():
        return None
    si = int(np.argmax(enter))           # 訊號分鐘
    ai = si + ENTRY_DELAY                 # 實際成交分鐘（延遲）
    if ai >= len(sp) or np.isnan(sp[ai]):
        return None
    S_in, t_in = sp[ai], ts[ai]
    post = slice(ai + 1, len(sp))
    sps, tss = sp[post], ts[post]
    if len(sps) == 0:
        return None
    v = ~np.isnan(sps)
    max_hold_ms = p["max_hold_h"] * 3_600_000
    conv = v & (sps <= p["exit_sp"])
    stop = v & (sps >= p["stop_sp"])
    tout = v & ((tss - t_in) >= max_hold_ms)
    cands = []
    if conv.any(): cands.append((int(np.argmax(conv)), "converge"))
    if stop.any(): cands.append((int(np.argmax(stop)), "stop"))
    if tout.any(): cands.append((int(np.argmax(tout)), "timeout"))
    if cands:
        k, reason = min(cands, key=lambda x: x[0])
    else:
        idx = np.where(v)[0]
        if len(idx) == 0:
            return None
        k, reason = int(idx[-1]), "end"
    S_out = sps[k]
    held_min = (tss[k] - t_in) / 60000
    mae = float(np.nanmax(sps[:k + 1])) - S_in if k >= 0 else 0.0

    xi = ai + 1 + k
    bn_in, bb_in = ev["bn_last"][ai], ev["bb_last"][ai]
    if not (bn_in == bn_in) or not (bb_in == bb_in):
        funding = 0.0
    else:
        fr_H, fr_L = (ev["bn_fr"], ev["bb_fr"]) if bn_in >= bb_in else (ev["bb_fr"], ev["bn_fr"])
        funding = (_settle_sum(fr_H, ai, xi) - _settle_sum(fr_L, ai, xi)) * 100

    pnl = (S_in - S_out) - cost4 + funding
    return {"symbol": ev["symbol"], "start_ms": ev["start_ms"], "pnl": pnl,
            "S_in": float(S_in), "S_out": float(S_out), "held_min": held_min,
            "reason": reason, "mae": max(0.0, mae), "funding": funding}


def run_combo(evs, p, cost4):
    return [t for t in (sim_event(ev, p, cost4) for ev in evs) if t]


def agg(trades):
    if not trades:
        return {"n_trades": 0, "total_pnl": 0.0}
    pnl = np.array([t["pnl"] for t in trades])
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    return {
        "n_trades": len(trades),
        "total_pnl": round(float(pnl.sum()), 2),
        "avg_pnl": round(float(pnl.mean()), 4),
        "median_pnl": round(float(np.median(pnl)), 4),
        "win_rate": round(float((pnl > 0).mean()), 4),
        "profit_factor": round(float(pf), 3) if pf != float("inf") else None,
        "avg_funding": round(float(np.mean([t["funding"] for t in trades])), 4),
        "max_drawdown": round(_max_dd(pnl), 2),
        "p99_mae": round(float(np.percentile([t["mae"] for t in trades], 99)), 2),
        "total_pnl_ex_top5pct": round(_pnl_ex_top(pnl, 0.05), 2),
        "converge_rate": round(float(np.mean([t["reason"] == "converge" for t in trades])), 4),
    }


def _max_dd(pnl):
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def _pnl_ex_top(pnl, frac):
    n = max(1, int(len(pnl) * frac))
    return float(np.sort(pnl)[:-n].sum()) if len(pnl) > n else 0.0


def neighborhood_score(scores: dict, key: tuple) -> float:
    """參數鄰域(各維度±1格)平均總報酬 → 高原指標（穩健）。"""
    idx = {k: GRID[k].index(v) for k, v in zip(GKEYS, key)}
    vals = []
    for offsets in product([-1, 0, 1], repeat=len(GKEYS)):
        nb = []
        ok = True
        for k, off in zip(GKEYS, offsets):
            j = idx[k] + off
            if not (0 <= j < len(GRID[k])):
                ok = False; break
            nb.append(GRID[k][j])
        if ok and tuple(nb) in scores:
            vals.append(scores[tuple(nb)])
    return float(np.mean(vals)) if vals else -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=0.05, help="每側 taker 手續費 %")
    ap.add_argument("--slip", type=float, default=0.10, help="每腿滑點 %")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    cost4 = 4 * (args.fee + args.slip)

    evs = load_events()
    evs.sort(key=lambda e: e["start_ms"])           # 按時間排序供切分
    n = len(evs)
    cut = int(n * (1 - OOS_FRAC))
    is_evs, oos_evs = evs[:cut], evs[cut:]
    print(f"[backtest v2] {n} 事件（IS {len(is_evs)} / OOS {len(oos_evs)}），"
          f"網格 {int(np.prod([len(v) for v in GRID.values()]))} 組，"
          f"進場延遲 {ENTRY_DELAY}min，成本 4×({args.fee}+{args.slip})={cost4:.2f}%")

    # 1) IS 跑全網格
    is_scores, is_aggs = {}, {}
    for combo in product(*GRID.values()):
        p = dict(zip(GKEYS, combo))
        a = agg(run_combo(is_evs, p, cost4))
        is_scores[combo] = a["total_pnl"]
        is_aggs[combo] = a

    # 2) 穩健選參：鄰域平均最高且自身正
    ranked = sorted(is_scores.keys(),
                    key=lambda c: neighborhood_score(is_scores, c), reverse=True)
    robust = next((c for c in ranked if is_scores[c] > 0
                   and is_aggs[c]["n_trades"] >= 20), None)
    peak = max(is_scores.keys(), key=lambda c: is_scores[c])

    # 3) OOS 驗證（穩健組 與 峰值組 都驗，對比）
    def describe(combo):
        p = dict(zip(GKEYS, combo))
        return (f"prem≥{p['premium_thr']} entry≥{p['entry_sp']} exit≤{p['exit_sp']} "
                f"stop≥{p['stop_sp']} hold≤{p['max_hold_h']}h")

    report = {"params_cost": {"fee": args.fee, "slip": args.slip, "entry_delay_min": ENTRY_DELAY},
              "n_events": n, "n_is": len(is_evs), "n_oos": len(oos_evs), "grid": GRID}
    for label, combo in (("robust", robust), ("peak", peak)):
        if combo is None:
            report[label] = None
            continue
        p = dict(zip(GKEYS, combo))
        is_a = is_aggs[combo]
        oos_a = agg(run_combo(oos_evs, p, cost4))
        full_a = agg(run_combo(evs, p, cost4))
        report[label] = {"desc": describe(combo), "params": p,
                         "neighborhood_avg": round(neighborhood_score(is_scores, combo), 1),
                         "in_sample": is_a, "out_sample": oos_a, "full": full_a}

    # 4) 全網格(全資料)排行供參考
    full_rank = []
    for combo in product(*GRID.values()):
        p = dict(zip(GKEYS, combo))
        a = agg(run_combo(evs, p, cost4))
        if a["n_trades"] > 0:
            full_rank.append({"desc": describe(combo), **a})
    full_rank.sort(key=lambda r: r["total_pnl"], reverse=True)
    n_pos = sum(1 for r in full_rank if r["total_pnl"] > 0)
    report["grid_summary"] = {"n_combos": len(full_rank), "n_positive": n_pos,
                              "pct_positive": round(n_pos / len(full_rank) * 100, 1)}
    report["full_rank_top"] = full_rank[:args.top]

    # 5) 溢價門檻「隔離對照」：其他參數固定，只掃 premium_thr，供前端 apples-to-apples 展示。
    #    基準取「較有耐心」的一組（停損寬、持有長）＝策略的相對有利情境。⚠ 絕對盈虧會隨
    #    出場/停損策略改變（換中庸或保守基準時 8% 也轉負），唯一跨配置與成本皆穩健的，是
    #    『門檻越低虧越凶、2%/5% 一律墊底』的單調排序——這才是「告警門檻該拉高」的依據，
    #    勿據此單表斷言某門檻必獲利。
    ISO_FIXED = {"entry_sp": 3.0, "exit_sp": 0.5, "stop_sp": 12.0, "max_hold_h": 24}
    report["premium_thr_isolation"] = {
        "fixed_params": ISO_FIXED,
        "by_threshold": [
            {"premium_thr": thr, **agg(run_combo(evs, {**ISO_FIXED, "premium_thr": thr}, cost4))}
            for thr in GRID["premium_thr"]
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 列印
    print(f"\n網格存活率（全資料正報酬組）：{n_pos}/{len(full_rank)} = {report['grid_summary']['pct_positive']}%")
    for label in ("robust", "peak"):
        r = report.get(label)
        tag = "穩健組(鄰域高原)" if label == "robust" else "峰值組(IS最高)"
        if not r:
            print(f"\n【{tag}】無（IS 無正報酬穩健組）")
            continue
        print(f"\n【{tag}】{r['desc']}  鄰域均={r['neighborhood_avg']}")
        for seg in ("in_sample", "out_sample", "full"):
            a = r[seg]
            print(f"  {seg:10} 筆={a.get('n_trades'):4} 總PnL={a.get('total_pnl'):8}% "
                  f"勝率={a.get('win_rate',0)*100:5.1f}% 均={a.get('avg_pnl')}% "
                  f"PF={a.get('profit_factor')} MDD={a.get('max_drawdown')}% "
                  f"去top5%後={a.get('total_pnl_ex_top5pct')}%")
    print(f"\n[backtest] 完成 → {OUT.name}")


if __name__ == "__main__":
    main()
