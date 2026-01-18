@echo off
setlocal

set ROOT=%~dp0

if exist "%ROOT%.venv\Scripts\python.exe" (
  set PY=%ROOT%.venv\Scripts\python.exe
) else (
  set PY=python
)

cd /d "%ROOT%"

echo Lancement OpenSpartan Graphs...
"%PY%" "%ROOT%run_dashboard.py"

echo.
echo (Ferme cette fenetre pour arreter le serveur)
pause
