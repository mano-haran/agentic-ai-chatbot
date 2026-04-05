#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="/Users/mano/Work/venv"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$APP_DIR/.pid"

echo ""
echo " ============================================================"
echo "  Agentic Framework - DevOps AI Assistant"
echo " ============================================================"
echo ""

# ── Validate .env ──────────────────────────────────────────────────────────────
if [[ ! -f "$APP_DIR/.env" ]]; then
    echo " [ERROR] .env file not found."
    echo "         Copy .env.example to .env and add your API keys:"
    echo ""
    echo "           cp .env.example .env"
    echo ""
    exit 1
fi

# ── Validate virtual environment ───────────────────────────────────────────────
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo " [ERROR] Virtual environment not found at: $VENV_PATH"
    echo "         Create it and install dependencies:"
    echo ""
    echo "           python3 -m venv $VENV_PATH"
    echo "           $VENV_PATH/bin/pip install -r requirements.txt"
    echo ""
    exit 1
fi

# ── No-proxy for local endpoints ───────────────────────────────────────────────
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1

# ── Load .env ──────────────────────────────────────────────────────────────────
set -a
# shellcheck disable=SC1091
source "$APP_DIR/.env"
set +a

# ── Activate virtual environment ───────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# ── Print startup summary ──────────────────────────────────────────────────────
echo " Provider : ${DEFAULT_PROVIDER:-openai}"
echo " Model    : ${DEFAULT_MODEL:-gpt-4o}"
echo " Routing  : ${DEFAULT_ROUTING_MODEL:-gpt-4o-mini}"
echo " NO_PROXY : $NO_PROXY"
echo ""
echo " Starting Chainlit on http://localhost:8000"
echo " PID file : $PID_FILE"
echo " Press Ctrl+C to stop."
echo ""

# ── Launch ─────────────────────────────────────────────────────────────────────
cd "$APP_DIR"
chainlit run app.py &
APP_PID=$!
echo $APP_PID > "$PID_FILE"
echo " Started with PID $APP_PID"
echo ""

wait $APP_PID
rm -f "$PID_FILE"
