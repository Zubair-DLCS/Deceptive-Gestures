@echo off
rem Study 1: one student at this laptop. Double-click to start.
cd /d "%~dp0"
python apps\session_app.py %*
pause
