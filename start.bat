@echo off
setlocal
title AutoMate AI
cd /d "%~dp0"

if exist ".venv\installed.flag" goto launch
echo First-time setup is needed - handing over to run.bat ...
call "%~dp0run.bat"
exit /b %errorlevel%

:launch
echo Starting AutoMate AI at http://localhost:8501
echo Keep this window open while you use it. Press Ctrl+C here to stop.
echo.
start "" /b cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
if not errorlevel 1 goto end
echo.
echo [ERROR] The app stopped with an error - see the message above.
echo         If it says port 8501 is already in use, close the other AutoMate AI window first.
pause
exit /b 1

:end
endlocal
