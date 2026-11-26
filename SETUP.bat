@echo off
REM One-time setup on a new Windows laptop. Needs Python 3.10-3.13 on PATH (python.org installer, tick "Add to PATH").
REM Creates .venv, installs everything (about 1-3 GB download), runs the tests and one demo prediction.
cd /d "%~dp0"
python --version || (echo Python not found. Install Python 3.13 from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo === Installing PyTorch (CPU build works everywhere; see README for NVIDIA GPU) ===
pip install torch torchvision || (echo PyTorch install failed & pause & exit /b 1)
echo === Installing project requirements + YOLO + dashboard ===
pip install -r requirements.txt ultralytics streamlit || (echo install failed & pause & exit /b 1)
REM ultralytics pulls in plain opencv-python; the HOG baseline needs the contrib build, and two cv2 packages clash.
pip uninstall -y opencv-python opencv-python-headless
pip install --force-reinstall --no-deps "opencv-contrib-python-headless==5.0.0.93" || pip install --force-reinstall --no-deps "opencv-contrib-python-headless<5.1"
set PYTHONPATH=src
echo === Running tests ===
python -m pytest -q -p no:cacheprovider tests
echo === Demo prediction on test_images\rural_signs.webp ===
python -m traffic_monitor.cli predict --source test_images\rural_signs.webp --output outputs\predictions\demo
echo.
echo Setup done. Next: drag any road photo onto run_on_image.bat, or double-click run_dashboard.bat
if not "%NOPAUSE%"=="1" pause
