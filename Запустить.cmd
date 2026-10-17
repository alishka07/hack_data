@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run first: py -3.11 setup_local.py
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launch_web.py
pause
