@echo off
setlocal
title AutoMate AI
cd /d "%~dp0"

echo ============================================================
echo   AutoMate AI - setup and launch
echo ============================================================
echo.

rem ---- 1. find Python ------------------------------------------------
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto have_python
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto have_python
echo [ERROR] Python was not found on this computer.
echo         Install Python 3.11 or newer from https://www.python.org/downloads/
echo         and tick "Add python.exe to PATH" during installation. Then run this file again.
goto fail

:have_python
rem ---- 2. create the virtual environment (first run only) ------------
if exist ".venv\Scripts\python.exe" goto have_venv
echo [1/3] Creating virtual environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 goto venv_failed
if not exist ".venv\Scripts\python.exe" goto venv_failed
goto install

:venv_failed
echo [ERROR] Could not create the virtual environment.
goto fail

:have_venv
if exist ".venv\installed.flag" goto launch

:install
rem ---- 3. install dependencies (first run only) ----------------------
echo [2/3] Installing dependencies - this takes a few minutes the first time ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if not errorlevel 1 goto install_ok
echo.
echo [WARNING] The full dependency list failed to install. Installing only what the app needs ...
".venv\Scripts\python.exe" -m pip install streamlit pandas numpy plotly networkx
if errorlevel 1 goto install_failed
echo [WARNING] Optional packages are missing: mlxtend, pytest, matplotlib, python-docx.
echo           The app works without them. Tests and report rebuilding need them.

:install_ok
echo ok> ".venv\installed.flag"
goto launch

:install_failed
echo [ERROR] Dependencies could not be installed. Check your internet connection and run this file again.
goto fail

:launch
rem ---- 4. start the app ----------------------------------------------
echo [3/3] Starting AutoMate AI ...
echo.
echo   The app will open in your browser at http://localhost:8501
echo   Keep this window open while you use it. Press Ctrl+C here to stop.
echo.
start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
if errorlevel 1 goto app_failed
goto end

:app_failed
echo.
echo [ERROR] The app stopped with an error - see the message above.
echo         If it says port 8501 is already in use, close the other AutoMate AI window first.
goto fail

:fail
echo.
pause
exit /b 1

:end
endlocal
