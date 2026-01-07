@echo off
setlocal

set "SRC=%~dp0"
set "DL=%USERPROFILE%\Downloads"
set "ROOT=%DL%\Photo Processing"
if not exist "%DL%" set "ROOT=%USERPROFILE%\Photo Processing"
set "SCRIPTS=%ROOT%\Automation Scripts"

mkdir "%SCRIPTS%" 2>nul

copy /Y "%SRC%auto_stamp_and_kmz.py" "%SCRIPTS%" >nul
copy /Y "%SRC%requirements.txt" "%SCRIPTS%" >nul
copy /Y "%SRC%RUN_ONE_CLICK.bat" "%SCRIPTS%" >nul
if exist "%SRC%python\python.exe" (
  xcopy /E /I /Y "%SRC%python" "%SCRIPTS%\python" >nul
)

pushd "%SCRIPTS%"
set "LOCAL_PY=%SCRIPTS%\python"
if exist "%LOCAL_PY%\pythonw.exe" (
  "%LOCAL_PY%\pythonw.exe" auto_stamp_and_kmz.py
  popd
  exit /b
)
if exist "%LOCAL_PY%\python.exe" (
  "%LOCAL_PY%\python.exe" auto_stamp_and_kmz.py
  popd
  exit /b
)
where pythonw >nul 2>&1
if %errorlevel%==0 (
  pythonw auto_stamp_and_kmz.py
  popd
  exit /b
)

where python >nul 2>&1
if %errorlevel%==0 (
  python auto_stamp_and_kmz.py
  popd
  exit /b
)

popd
echo Python was not found.
echo Install Python from the Microsoft Store or enable App Execution Aliases.
echo Opening Microsoft Store...
start "" "ms-windows-store://pdp/?productid=9NRWMJP3717K"
start "" "https://www.microsoft.com/store/apps/9NRWMJP3717K"
pause
