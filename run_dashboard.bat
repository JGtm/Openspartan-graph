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
set DEFAULT_GAMERTAG=JGtm
set DEFAULT_XUID=2533274823110022

if "%SPNKR_PLAYER%"=="" (
  set SPNKR_PLAYER=%DEFAULT_GAMERTAG%
)

echo.
echo [SPNKr] Joueur cible: %SPNKR_PLAYER%
echo [SPNKr] (Default gamertag=%DEFAULT_GAMERTAG% / xuid=%DEFAULT_XUID%)
echo [SPNKr] (Override possible via variable d'env SPNKR_PLAYER)

echo.
echo [SPNKr] Refresh desactive (mode lecture seule)

rem Auto-sélection de la DB SPNKr (priorité: data\spnkr_gt_<player>.db, puis data\spnkr_<player>.db, sinon plus récente)
set "SPNKR_DB="
set "CAND1=%ROOT%data\spnkr_gt_%SPNKR_PLAYER%.db"
if exist "%CAND1%" set "SPNKR_DB=%CAND1%"

if "%SPNKR_DB%"=="" (
  set "CAND2=%ROOT%data\spnkr_%SPNKR_PLAYER%.db"
  if exist "%CAND2%" set "SPNKR_DB=%CAND2%"
)

if "%SPNKR_DB%"=="" (
  for /f "delims=" %%D in ('dir /b /a:-d /o:-d "%ROOT%data\spnkr*.db" 2^>nul') do (
    set "SPNKR_DB=%ROOT%data\%%D"
    goto :db_found
  )
)

:db_found
if not "%SPNKR_DB%"=="" (
  set "OPENSPARTAN_DB_PATH=%SPNKR_DB%"
  echo [DB] Auto-select: %OPENSPARTAN_DB_PATH%
) else (
  echo [DB] Aucune DB SPNKr trouvee dans %ROOT%data\spnkr*.db
)

rem IMPORTANT: on ne lance aucun refresh automatique pour le moment. 
rem Si plusieurs DB SPNKr existent (spnkr_*.db), on laisse la sidebar choisir.
rem OPENSPARTAN_DB_PATH n'est donc plus force ici.
rem Ajouter --refresh-spnkr pour un refresh auto.
"%PY%" "%ROOT%run_dashboard.py"

echo.
echo (Ferme cette fenetre pour arreter le serveur)
pause
