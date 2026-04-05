#!/usr/bin/env bash

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$APP_DIR/.pid"
FOUND=0

echo ""
echo " ============================================================"
echo "  Agentic Framework - Stopping DevOps AI Assistant"
echo " ============================================================"
echo ""

# ── Strategy 1: kill by PID file ──────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    APP_PID=$(cat "$PID_FILE")
    if [[ -n "$APP_PID" ]]; then
        echo " Stopping process PID $APP_PID ..."
        if kill -0 "$APP_PID" 2>/dev/null; then
            kill -TERM "$APP_PID" 2>/dev/null
            sleep 1
            # Force kill if still running
            if kill -0 "$APP_PID" 2>/dev/null; then
                kill -KILL "$APP_PID" 2>/dev/null
            fi
            echo " Stopped."
        else
            echo " PID $APP_PID not found (may have already stopped)."
        fi
        rm -f "$PID_FILE"
        FOUND=1
    fi
fi

# ── Strategy 2: find by process name (fallback when no .pid file) ─────────────
if [[ "$FOUND" -eq 0 ]]; then
    echo " No .pid file found. Searching for running Chainlit instance..."
    echo ""

    PIDS=$(pgrep -f "chainlit run app.py" 2>/dev/null || true)

    if [[ -n "$PIDS" ]]; then
        for PID in $PIDS; do
            echo " Stopping PID $PID ..."
            kill -TERM "$PID" 2>/dev/null || true
        done
        sleep 1
        # Force kill any survivors
        for PID in $PIDS; do
            if kill -0 "$PID" 2>/dev/null; then
                kill -KILL "$PID" 2>/dev/null || true
                echo " Force-killed PID $PID"
            fi
        done
        FOUND=1
    else
        echo " No running Chainlit instance found."
    fi
fi

echo ""
echo " Done."
echo ""
