@echo off
cd /d "%~dp0.."
python offline_tools\visualize_circle_detection.py
if errorlevel 1 pause
