@echo off
REM One-click sync: Google Sheets -> CSV -> JSON
REM Edit scripts/sync_from_sheets.py (SHEET_ID, SHEET_MAP) to match your sheet.
cd /d "%~dp0"
python scripts/sync_from_sheets.py --fetch
pause
