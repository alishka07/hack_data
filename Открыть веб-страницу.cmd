@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" launch_web.py
) else (
  python launch_web.py
)
if errorlevel 1 pause
