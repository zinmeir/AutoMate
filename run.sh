#!/usr/bin/env bash
# macOS / Linux equivalent of run.bat:  bash run.sh
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[1/3] Creating virtual environment ..."
  python3 -m venv .venv
fi
if [ ! -f .venv/installed.flag ]; then
  echo "[2/3] Installing dependencies ..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt || {
    echo "[WARNING] Full install failed; installing only what the app needs ..."
    .venv/bin/python -m pip install streamlit pandas numpy plotly networkx
  }
  echo ok > .venv/installed.flag
fi
echo "[3/3] Starting AutoMate AI at http://localhost:8501 (Ctrl+C to stop)"
exec .venv/bin/python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
