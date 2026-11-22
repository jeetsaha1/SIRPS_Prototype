@echo off
title SIRPS Web Application
echo ========================================================
echo   Starting SIRPS Interactive Web Application...
echo   Opening website in your browser at http://localhost:8501
echo ========================================================

cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
