@echo off
setlocal EnableDelayedExpansion

echo.
echo  ============================================================
echo   Agentic Framework - DevOps AI Assistant
echo  ============================================================
echo.

:: ── No-proxy for local endpoints ──────────────────────────────────────────────
::
:: Prevents HTTP proxy settings from intercepting requests to:
::   - localhost / 127.0.0.1  : local APIs (Ollama, vLLM, LM Studio)
::   - 0.0.0.0                : wildcard local bind
::   - ::1                    : IPv6 loopback
::
:: Both variables are set because different tools check different names.
::
set NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1
set no_proxy=localhost,127.0.0.1,0.0.0.0,::1

:: ── Validate .env ──────────────────────────────────────────────────────────────
if not exist .env (
    echo [ERROR] .env file not found.
    echo         Copy .env.example to .env and add your API keys:
    echo.
    echo           copy .env.example .env
    echo.
    pause
    exit /b 1
)

:: ── Load variables from .env ───────────────────────────────────────────────────
::
:: eol=# skips comment lines (lines starting with #).
:: tokens=1,* delims==  puts the key in %%A and the full value (including any
:: embedded = signs such as base64 keys) in %%B.
::
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" (
        set "%%A=%%B"
    )
)

:: ── Validate virtual environment ───────────────────────────────────────────────
if not exist .venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found.
    echo         Create it and install dependencies:
    echo.
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── Activate virtual environment ───────────────────────────────────────────────
call .venv\Scripts\activate.bat

:: ── Print startup summary ──────────────────────────────────────────────────────
echo  Provider : %DEFAULT_PROVIDER%
echo  Model    : %DEFAULT_MODEL%
echo  Routing  : %DEFAULT_ROUTING_MODEL%
echo  NO_PROXY : %NO_PROXY%
echo.
echo  Starting Chainlit on http://localhost:8000
echo  Press Ctrl+C to stop.
echo.

:: ── Launch ────────────────────────────────────────────────────────────────────
::
:: Runs in the foreground so logs are visible in this window.
:: Ctrl+C will cleanly stop the server.
::
chainlit run app.py

endlocal
