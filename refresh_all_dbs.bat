@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================================
rem Refresh toutes les DB SPNKr du dossier data/ avec "toutes les options".
rem
rem - Utilise le script: scripts\spnkr_import_db.py
rem - Active: skill + assets + highlight events
rem - Mode: --resume (backfill / incrémental)
rem
rem Usage:
rem   refresh_all_dbs.bat [max_matches] [match_type] [requests_per_second]
rem
rem Exemples:
rem   refresh_all_dbs.bat
rem   refresh_all_dbs.bat 200 matchmaking 5
rem   refresh_all_dbs.bat 400 all 3
rem
rem Pré-requis:
rem   - Tokens dans .env.local (SPNKR_SPARTAN_TOKEN / SPNKR_CLEARANCE_TOKEN)
rem     ou bien config Azure (SPNKR_AZURE_CLIENT_ID/SECRET + SPNKR_OAUTH_REFRESH_TOKEN)
rem ============================================================================

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

set "MAX_MATCHES=%~1"
if not defined MAX_MATCHES set "MAX_MATCHES=200"

set "MATCH_TYPE=%~2"
if not defined MATCH_TYPE set "MATCH_TYPE=matchmaking"

set "RPS=%~3"
if not defined RPS set "RPS=5"

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if not exist "scripts\spnkr_import_db.py" (
  echo [ERREUR] scripts\spnkr_import_db.py introuvable. Lance ce .bat depuis la racine du repo.
  popd >nul
  exit /b 2
)

if not exist "data" (
  echo [ERREUR] Dossier data/ introuvable.
  popd >nul
  exit /b 2
)

echo ==============================================
echo Refresh DBs SPNKr
echo - root: %CD%
echo - python: %PY%
echo - max_matches: %MAX_MATCHES%
echo - match_type: %MATCH_TYPE%
echo - requests_per_second: %RPS%
echo ==============================================

set "FOUND=0"

for %%F in ("%ROOT%data\spnkr*.db") do (
  set "FOUND=1"
  set "DB=%%~fF"
  set "BASE=%%~nF"

  set "PLAYER="
  if /I "!BASE:~0,9!"=="spnkr_gt_" (
    set "PLAYER=!BASE:~9!"
  ) else if /I "!BASE:~0,11!"=="spnkr_xuid_" (
    set "PLAYER=!BASE:~11!"
  ) else if /I "!BASE:~0,6!"=="spnkr_" (
    set "PLAYER=!BASE:~6!"
  )

  if not defined PLAYER (
    echo.
    echo [SKIP] %%~nxF ^(impossible de déduire --player depuis le nom^)
    echo        Conseil: renomme en spnkr_gt_^<Gamertag^>.db ou spnkr_xuid_^<XUID^>.db
  ) else (
    echo.
    echo --------------------------------------------------
    echo [DB] !DB!
    echo [PLAYER] !PLAYER!
    echo --------------------------------------------------

    "%PY%" scripts\spnkr_import_db.py ^
      --out-db "!DB!" ^
      --player "!PLAYER!" ^
      --match-type "%MATCH_TYPE%" ^
      --max-matches %MAX_MATCHES% ^
      --requests-per-second %RPS% ^
      --resume ^
      --with-highlight-events

    if errorlevel 1 (
        echo [ERREUR] Refresh en échec pour %%~nxF (code=!errorlevel!^).
      echo         Je continue avec la DB suivante...
    ) else (
      echo [OK] %%~nxF
    )
  )
)

if "%FOUND%"=="0" (
  echo [INFO] Aucune DB trouvée: data\spnkr*.db
)

echo.
echo Terminé.
popd >nul
endlocal
exit /b 0
