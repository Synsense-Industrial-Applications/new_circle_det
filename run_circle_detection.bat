@echo off
cd /d "%~dp0"
python visualize_circle_detection.py
if errorlevel 1 pause
