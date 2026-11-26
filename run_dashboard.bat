@echo off
REM Starts the Streamlit dashboard in your browser. Close this window to stop it.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -c "import streamlit" 2>nul || pip install streamlit
set PYTHONPATH=src
echo Starting dashboard... your browser will open at http://localhost:8501
python -m streamlit run app\dashboard.py
pause
