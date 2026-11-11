@echo off
rem Try the whole session without a camera or videos (simulated face numbers).
cd /d "%~dp0"
python apps\session_app.py --dry-run --data data_dryrun %*
pause
