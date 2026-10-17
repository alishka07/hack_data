@echo off
cd /d "%~dp0"
if exist "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe" (
  "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe" launch_web.py
) else (
  python launch_web.py
)
if errorlevel 1 pause
