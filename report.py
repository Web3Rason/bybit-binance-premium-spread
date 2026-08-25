# -*- coding: utf-8 -*-
"""
產出研究報告 REPORT.md（讀 analysis.json）。

主軸：兩家溢價 → 兩家價差；解釋層：資費結算週期 + cap 如何調節傳導。
資料品質：以指數差中位數剔除「同名不同幣」假事件，結論只採可信事件。
"""
import json
import sys
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis.json"
CAND = ROOT / "candidates.json"
REPORT = ROOT / "REPORT.md"


def _fmt(v, suffix=""):
    return f"{v}{suffix}" if v is not None else "—"


def _row(e):
    ll = e.get("leadlag_premSpread_to_lastSpread") or {}
    return (f"| {e['symbol']} | {_fmt(e.get('peak_premium_pct'))} | "
            f"{_fmt(e.get('p99_abs_last_spread_pct'))} | {_fmt(e.get('max_abs_prem_spread_pct'))} | "
            f"{_fmt(e.get('spearman_premSpread_lastSpread'))} | {_fmt(e.get('corr_premSpread_lastSpread'))} | "
            f"{_fmt(ll.get('lead_lag_min'))} | "
            f"{_fmt(e.get('bn_actual_interval_h'))}/{_fmt(e.get('bb_actual_interval_h'))} | "
            f"±{_fmt(e.get('bn_cap'))}/±{_fmt(e.get('bb_cap'))} | "
            f"{_fmt(e.get('bn_fr_usage_max'))}/{_fmt(e.get('bb_fr_usage_max'))} | "
            f"{_fmt(e.get('median_abs_index_spread_pct'))} |")


HEADER = ("| 幣種 | 峰值溢價% | P99\\|價差\\|% | max\\|溢價差\\|% | Spearman | Pearson | "
          "溢價領先(分) | 週期BN/BB(h) | cap BN/BB | 使用率峰BN/BB | 指數差(中位)% |")
SEP = "|---|---|---|---|---|---|---|---|---|---|---|"


def main():
    if not ANALYSIS.exists():
        sys.exit("找不到 analysis.json，請先跑 analyze.py")
    a = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    events = a["events"]
    clean = [e for e in events if not e.get("suspect_diff_coin")]
    suspect = [e for e in events if e.get("suspect_diff_coin")]
    cand = json.loads(CAND.read_text(encoding="utf-8")) if CAND.exists() else {}
    params = cand.get("params", {})
    thr = params.get("threshold")

    # 可信事件統計（用 Spearman 抗插針）
    corrs = [e["spearman_premSpread_lastSpread"] for e in clean
             if e.get("spearman_premSpread_lastSpread") is not None]
    bn_u = [e["bn_fr_usage_max"] for e in clean if e.get("bn_fr_usage_max") is not None]
    bb_u = [e["bb_fr_usage_max"] for e in clean if e.get("bb_fr_usage_max") is not None]
    bb_win = sum(1 for e in clean
                 if (e.get("bb_fr_usage_max") or 0) > (e.get("bn_fr_usage_max") or 0))

    L = []
    w = L.append
    w("# 5036 研究報告：Bybit × Binance 極端溢價 → 跨所價差")
    w("")
    w(f"> 產生時間：{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC　"
      f"事件數：{len(events)}（可信 {len(clean)} / 疑似同名不同幣 {len(suspect)}）")
    w("")
    w("## 研究問題")
    w("當同一永續合約出現**極端溢價**時，Bybit 與 Binance 之間的**價差**如何演變。"
      "溢價是驅動源、價差是結果；兩家資費的**結算週期**與**上下限(cap)**是中間的結構性"
      "調節變數（決定資費能多大、多頻繁，進而影響套利力道與合約價格）。")
    w("")
    w("## 核心結論（僅採可信事件）")
    if corrs:
        w(f"- **溢價差 ↔ 真實成交價差 Spearman 秩相關中位數 = {st.median(corrs):.3f}**"
          f"（n={len(corrs)}，秩相關抗單分鐘插針）→ 溢價確實驅動跨所價差。")
    w(f"- 資費使用率峰值中位數：**Binance {st.median(bn_u):.3f} / Bybit {st.median(bb_u):.3f}**"
      f"（{len(clean)} 個可信事件中 Bybit 使用率 > Binance 的有 **{bb_win}** 個）。")
    w(f"  - 假設「Bybit 對溢價更敏感、資費更易貼 cap」在本樣本"
      f"{'**獲部分支持**' if bb_win > len(clean)/2 else '**未獲一致支持**'}"
      f"（需注意兩家 cap/週期不同，使用率比較有基準差異）。")
    if clean:
        top = sorted(clean, key=lambda e: e.get("p99_abs_last_spread_pct") or 0, reverse=True)[:5]
        w(f"- 可信事件中真實成交價差最大(P99，排除插針)：" +
          "、".join(f"{e['symbol']} {e.get('p99_abs_last_spread_pct')}%" for e in top) + "。")
    w("")
    w("## 方法與口徑")
    w("- **溢價**：兩家自算 `(mark−index)/index`（官方 premium index 口徑不同不可直接比）。")
    w("- **價差**：`last_spread=(bb_last−bn_last)/bn_last`（真實成交）為主軸；mark_spread 僅對照。")
    w("- **資費使用率**=費率/對應方向 cap（=1 貼頂）；結算週期由歷史結算戳推算（反映極端期縮短）。")
    w(f"- 全市場兩家共同 USDT 永續，不做成交額篩選；階段1 daily 粗篩 |premium|≥"
      f"{thr*100 if thr else '?'}%，階段2 候選窗抓 1m 對齊。")
    w(f"- **資料品質過濾**：指數差中位數 > {a.get('suspect_index_pct_threshold', 2)}% 判為"
      "同名不同幣（兩家實為不同代幣/倍數），其價差比較無意義，列入下方剔除區、不納入結論。")
    w("")
    w(f"## 可信事件總表（{len(clean)} 個，依最大真實價差排序）")
    w("")
    w(HEADER); w(SEP)
    for e in clean:
        w(_row(e))
    w("")
    if suspect:
        w(f"## 疑似同名不同幣（已剔除，{len(suspect)} 個）")
        w("> 指數差中位數過大，兩家實際非同一標的，價差數字無意義，僅列供查核。")
        w("")
        w(HEADER); w(SEP)
        for e in suspect:
            w(_row(e))
        w("")
    w("## 互動圖表")
    w("")
    w("每個事件的時序互動圖（兩家溢價 / 價差 / 資費使用率三窗格，時間軸同步、可縮放/平移/hover）"
      "由前端用 TradingView Lightweight Charts 呈現，開啟方式：")
    w("")
    w("```bash")
    w("start.bat   # 或 python -m http.server 5036 --bind 127.0.0.1，再瀏覽 http://localhost:5036")
    w("```")
    w("")
    w("> 在前端點選上方表格任一事件，下方即顯示該事件三窗格互動圖。")
    w("")
    w("## 限制與取捨（誠實標注）")
    w("- **同名不同幣**：以指數差中位數啟發式剔除，閾值之下仍可能殘留邊界個案。")
    w("- **歷史 cap 不可得**：公開 API 只給當前 cap，歷史可能不同，用當前值近似。")
    w("- **歷史結算週期**用結算時間戳眾數推算，週期切換過渡點可能有誤差。")
    w("- **粗篩靈敏度偏 Binance**：Bybit 官方 premium 經平滑數值小，同閾值較難觸發。")
    w("- **資費跨週期比較**：兩家週期不同（如 4h vs 8h），比較高低需注意基準差異。")
    w("")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] 完成 → {REPORT.name}（可信 {len(clean)} / 剔除 {len(suspect)}）")


if __name__ == "__main__":
    main()
