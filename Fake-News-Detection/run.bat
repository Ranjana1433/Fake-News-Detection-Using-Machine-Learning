@echo off
cd /d "%~dp0backend"
python -m pip install -r requirements.txt
python main.py
pause
