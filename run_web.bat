@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo Grid Splitter v4 - Web UI
echo ------------------------------
echo Open in browser: http://127.0.0.1:8765
echo Stop:  Ctrl+C
echo.
python -m uvicorn web.app:app --host 127.0.0.1 --port 8765
pause
