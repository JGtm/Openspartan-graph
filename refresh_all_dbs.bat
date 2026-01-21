@echo off
setlocal

rem [DEPRECATED] Utilise plutot le lanceur Python.
rem Exemple:
rem   .venv\Scripts\python openspartan_launcher.py refresh-all --max-matches 200 --match-type matchmaking --rps 5 --with-highlight-events

set ROOT=%~dp0
if exist "%ROOT%.venv\Scripts\python.exe" (
  set PY=%ROOT%.venv\Scripts\python.exe
) else (
  set PY=python
)

cd /d "%ROOT%"

set MAX_MATCHES=%~1
if "%MAX_MATCHES%"=="" set MAX_MATCHES=200
set MATCH_TYPE=%~2
if "%MATCH_TYPE%"=="" set MATCH_TYPE=matchmaking
set RPS=%~3
if "%RPS%"=="" set RPS=5

echo.
echo [INFO] refresh_all_dbs.bat est deprecated.
echo        Utilise plutot: %PY% openspartan_launcher.py refresh-all
echo.

"%PY%" "%ROOT%openspartan_launcher.py" refresh-all --max-matches %MAX_MATCHES% --match-type %MATCH_TYPE% --rps %RPS% --with-highlight-events

endlocal
exit /b %errorlevel%
