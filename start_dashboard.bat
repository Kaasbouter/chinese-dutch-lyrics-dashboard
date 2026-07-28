@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if errorlevel 1 goto :setup_failed

set "python_command="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "python_command=py -3"
if not defined python_command (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 set "python_command=python"
)
if not defined python_command goto :python_unavailable

if not exist ".venv\Scripts\python.exe" (
  echo Creating the private local Python environment...
  %python_command% -m venv .venv
  if errorlevel 1 goto :setup_failed
)

if exist ".venv\.lyrics-dashboard-ready" goto :dependencies_ready

echo Installing or checking the free local dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :setup_failed

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :setup_failed

".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :setup_failed

> ".venv\.lyrics-dashboard-ready" echo setup complete

:dependencies_ready
echo.
echo Starting the Chinese-Dutch Lyrics Converter...
echo Keep this window open while using the dashboard.
echo Close this window or press Ctrl+C to stop it.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
set "dashboard_exit=%errorlevel%"
endlocal & exit /b %dashboard_exit%

:python_unavailable
echo.
echo A compatible Python installation was not found.
echo Install Python 3.10 or newer from https://www.python.org/downloads/
echo During installation, enable "Add Python to PATH", then run this file again.
pause
endlocal
exit /b 1

:setup_failed
echo.
echo Dashboard setup failed. Review the error above.
echo The first setup requires an internet connection to download the free dependencies.
pause
endlocal
exit /b 1
