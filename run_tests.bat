@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\installed.flag" goto run
echo Run run.bat once first to set up the environment.
pause
exit /b 1

:run
".venv\Scripts\python.exe" -m pytest -q
echo.
echo Copy the summary line above into Section 12 of the report.
pause
endlocal
