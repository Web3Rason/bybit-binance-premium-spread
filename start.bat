@echo off
REM Bybit x Binance 極端溢價 -> 價差研究：用 HTTP server 服務 index.html
REM 只綁 127.0.0.1，避免把本機的快取與掃描結果整包對區網開放。
cd /d "%~dp0"
python -m http.server 5036 --bind 127.0.0.1
