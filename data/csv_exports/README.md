# CSV Exports for Google Sheets

These CSVs bridge Bender JSON data and Google Sheets for easy editing.

## Quick Sync (One Click or One Command)

**Update Bender from your Google Sheet:**

```
# Option A: Fetch from Sheets + convert to JSON
python scripts/sync_from_sheets.py --fetch

# Option B: Use local CSVs (after you download from Sheets)
python scripts/sync_from_sheets.py --local

# Windows: Double-click run_sync.bat (fetch) or run_sync_local.bat (local)
```

See `scripts/sync_from_sheets.py` — set `SHEET_ID` and `SHEET_MAP` to your sheet. Sheet must be shared as **Anyone with the link can view** for `--fetch` to work.

**Export JSON → CSV (for Sheets import):**

```
python scripts/json_to_csv_export.py
```

## Files

| CSV | Source JSON | Rows |
|-----|-------------|------|
| movement.csv | movement.json | 38 |
| speed_agility.csv | speed_agility.json | 1 |
| skating_mechanics.csv | skating_mechanics.json | 2 |
| warmup.csv | warmup.json | 18 |
| energy_systems.csv | energy_systems.json | 20 |
| stickhandling.csv | stickhandling.json | 45 |
| shooting.csv | shooting.json | 25 |
| mobility.csv | mobility.json | 24 |
| performance.csv | performance.json | 86 |
| circuits.csv | circuits.json | 8 |
| nhl_combine_results.csv | nhl_combine/nhl_combine_results.json | 372 |

## Import into Google Sheets

1. Create a new Google Sheet (or open existing).
2. File → Import → Upload → choose CSV.
3. Import location: **Replace spreadsheet** or **Insert new sheet(s)**.
4. Import one CSV per sheet tab (or create multiple sheets, one per CSV).

## Regenerating CSVs (JSON → CSV)

From the Bender repo root:

```
python scripts/json_to_csv_export.py
```

## Notes

- **circuits.csv**: `format` and `drills` columns contain JSON strings. Don’t break the JSON syntax when editing in Sheets.
- **nhl_combine_results.csv**: Sparse data — each player has only the tests they placed in.
