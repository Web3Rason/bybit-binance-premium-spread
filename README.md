# 5036 · Bybit × Binance 極端溢價 → 跨所價差研究

研究：**當同一永續合約出現極端溢價時，Bybit 與 Binance 之間的價差如何演變。**

溢價是驅動源、價差是結果；兩家資費的**結算週期**與**上下限(cap)**是中間的結構性
調節變數——它們決定資費能多大、收多頻繁，進而影響套利力道與兩家合約價格，造成價差。

## 核心結論用的指標口徑

| 名稱 | 定義 | 說明 |
|---|---|---|
| 溢價 `bn_prem`/`bb_prem` | `(mark − index) / index` | 兩家**自算**。官方 premium index K 線口徑不同（Binance 留盤中尖峰、Bybit 平滑），不可直接比 |
| 溢價差 `prem_spread` | `bb_prem − bn_prem` | 驅動源 |
| **價差** `last_spread` | `(bb_last − bn_last)/bn_last` | **主結果軸**：真實成交價差 |
| 標記價差 `mark_spread` | `(bb_mark − bn_mark)/bn_mark` | 僅對照（≈ index_spread + prem_spread，與溢價差部分恆等） |
| 指數差 `index_spread` | `(bb_index − bn_index)/bn_index` | 排除現貨指數取樣干擾 |
| 資費使用率 `fr_usage` | `rate / 對應方向 cap` | =1 表示貼頂封死（解釋層） |

## 資料來源（皆交易所原生 REST，禁用 ccxt）

- **Binance USDⓈ-M**：`premiumIndexKlines`/`markPriceKlines`/`indexPriceKlines`/`klines`、
  `fundingRate`、`fundingInfo`(cap/週期)、`exchangeInfo`
- **Bybit V5**：`premium-index-price-kline`/`mark-price-kline`/`index-price-kline`/`kline`、
  `funding/history`、`instruments-info`(cap/週期)
- 1m premium 歷史深度（實測）：Binance 3 年+、Bybit 2 年+。
- ⚠ Coinalyze 的 1min 只保留約 2 天且無 premium，不適合本研究，故不採用。

## 流程（兩階段漏斗，守機器負載規則：單程序序列）

```
scan.py        # 階段1 全市場粗篩：daily premium 找極端事件窗 → candidates.json
fetch_event.py # 階段2 候選窗深抓兩家 1m + funding，分鐘對齊 → events/*.parquet + events_index.json
analyze.py     # 主軸 溢價→價差：時序圖 + 相關 + 領先滯後 → charts/*.png + analysis.json
report.py      # 彙整 → REPORT.md
start.bat      # python -m http.server 5036，瀏覽 index.html 互動看圖
```

不做成交額/流動性篩選——只要溢價有極端就納入。

## 用法

```bash
pip install -r requirements.txt        # pandas / numpy / pyarrow
python scan.py                         # 全市場粗篩（預設回溯 4 年，API 自然截斷）
python scan.py --limit 20 --days 365   # 測試：前 20 個、近 1 年
python fetch_event.py --top-symbols 30 # 深抓峰值最大的前 30 個候選幣
python fetch_event.py --only TRBUSDT   # 只跑指定幣
python analyze.py
python report.py
start.bat                              # 開瀏覽器看 http://localhost:5036
```

## 限制（誠實標注）

- **歷史 cap 不可得**：公開 API 只給當前 cap，歷史可能不同，分析用當前值近似。
- **歷史結算週期**用結算時間戳眾數推算，週期切換過渡點可能有誤差。
- **粗篩靈敏度偏 Binance**：Bybit 官方 premium 經平滑數值小，同閾值較難觸發；
  「僅 Bybit 端極端」的罕見事件可能漏抓。
- **跨週期資費比較**：兩家週期不同（如 4h vs 8h），比較高低需注意基準差異。

圖表使用 [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts) v4.1.3（Apache-2.0）。
