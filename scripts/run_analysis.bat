@echo off
cd /d "%~dp0.."
python offline_tools\visualize_circle_detection.py --analyze --analysis-stride 50 --no-show
pause
