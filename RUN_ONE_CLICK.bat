@echo off
setlocal
cd /d "%~dp0"

where pythonw >nul 2>&1
if %errorlevel%==0 (
  pythonw auto_stamp_and_kmz.py
  exit /b
)

python auto_stamp_and_kmz.py
endlocal
