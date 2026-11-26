@echo off
REM Drag and drop any road image (jpg/png) onto this file to run the detector on it.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PYTHONPATH=src
if "%~1"=="" (
  echo Drag a road image onto run_on_image.bat, or run: run_on_image.bat path\to\image.jpg
  pause
  exit /b 1
)
python -m traffic_monitor.cli predict --source "%~1" --output outputs\predictions\my_images
start "" "%~dp0outputs\predictions\my_images\%~n1_annotated.jpg"
pause
