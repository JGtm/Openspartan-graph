@echo off
setlocal

set ROOT=%~dp0

if exist "%ROOT%.venv\Scripts\python.exe" (
  set PY=%ROOT%.venv\Scripts\python.exe
) else (
  set PY=python
)

cd /d "%ROOT%"

echo.
echo [INFO] run_dashboard.bat est deprecated.
echo        Utilise plutot: %PY% openspartan_launcher.py
echo.

"%PY%" "%ROOT%openspartan_launcher.py" %*

endlocal
exit /b %errorlevel%
