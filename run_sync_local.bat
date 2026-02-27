@echo off
REM Sync from local CSVs only (after manually downloading from Google Sheets)
cd /d "%~dp0"
python scripts/sync_from_sheets.py --local
pause
