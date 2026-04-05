@echo off
setlocal

echo.
echo  ============================================================
echo   Agentic Framework - Stopping DevOps AI Assistant
echo  ============================================================
echo.

set FOUND=0

:: ── Strategy 1: kill by PID file (if start.bat was launched via PowerShell) ───
::
:: start.bat writes a .pid file when launched in background mode.
:: This is the cleanest kill because it targets exactly the right process.
::
if exist .pid (
    set /p APP_PID=<.pid
    if not "!APP_PID!"=="" (
        echo  Stopping process PID !APP_PID! ...
        taskkill /PID !APP_PID! /T /F >nul 2>&1
        if !ERRORLEVEL! == 0 (
            echo  Stopped.
            set FOUND=1
        ) else (
            echo  PID !APP_PID! not found ^(may have already stopped^).
            set FOUND=1
        )
        del .pid >nul 2>&1
    )
)

:: ── Strategy 2: find by command line (fallback when no .pid file) ─────────────
::
:: Uses PowerShell + CIM to inspect the full command line of every python.exe
:: process and kills any that contain "chainlit" or "app.py".
::
:: Win32_Process is used instead of the deprecated WMIC.
::
if "!FOUND!"=="0" (
    echo  No .pid file found. Searching for running Chainlit instance...
    echo.

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Get-CimInstance Win32_Process -Filter 'Name = \"python.exe\"'" ^
        "| Where-Object { $_.CommandLine -like '*chainlit*' -or $_.CommandLine -like '*app.py*' }" ^
        "| ForEach-Object {" ^
        "    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue;" ^
        "    Write-Host ' Stopped PID' $_.ProcessId '-' $_.CommandLine.Substring(0, [Math]::Min(80,$_.CommandLine.Length))" ^
        "}" ^
        "-OutVariable procs" ^
        "; if (-not $procs) { Write-Host ' No running instance found.' }"

    if !ERRORLEVEL! == 0 (
        set FOUND=1
    )
)

echo.
echo  Done.
echo.

endlocal
