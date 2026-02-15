# NHL Combine / Fitness Testing Data

Shared dataset for **Bender** and **MotionApp**: NHL Scouting Combine–style results (draft-eligible prospects). **Years: 2015–2019, 2022–2025** (2020–2021 cancelled). Top-10 per test per year from topendsports.com, Bleacher Report, NHL.com. Use in Bender for benchmarks or workout context; use in MotionApp for grading or comparison.

## Schema (per player)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Player name (as at combine) |
| `year` | int | Combine year (e.g. 2025) |
| `height_in` | float? | Standing height (inches) |
| `wingspan_in` | float? | Wingspan (inches) |
| `standing_long_jump_in` | float? | Standing long jump / horizontal jump (inches) |
| `vertical_jump_in` | float? | Vertical jump (inches) |
| `vertical_jump_force_plate_in` | float? | Vertical jump, force plate (e.g. no-arm jump, inches) |
| `pull_ups` | int? | Max consecutive pull-ups |
| `bench_press` | mixed? | Bench press: reps at 50% BW, or `bench_power_watts_per_kg` in raw data |
| `bench_power_watts_per_kg` | float? | Bench press 50% BW – power (watts/kg), when not reported as reps |
| `pro_agility_left_sec` | float? | Pro agility 5-10-5, start left (seconds) |
| `pro_agility_right_sec` | float? | Pro agility 5-10-5, start right (seconds) |
| `y_balance_composite_cm` | float? | Y balance test composite (cm), when available |
| `source` | string? | e.g. `nhl_combine_2025`, `bleacher_report_top10` |

- All measurement fields are optional (null when not reported).
- File: `nhl_combine_results.json` — array of player objects.

## Sources

- **NHL**: [NHL Scouting Combine fitness testing results](https://www.nhl.com/news/scouting-combine-fitness-testing-results-289759684) (links to PDFs by year).
- **2025 PDF**: [2025 Scouting Combine Fitness Testing Results](https://media.nhl.com/site/vasset/public/attachments/2025/06/19060/2025ScoutingCombine_FitnessTestingResults_060725_FINAL.pdf) (top 25 per test).
- Top-10 summaries (e.g. Bleacher Report) are merged into this dataset where full PDF isn’t parsed.

## Usage

- **Bender**: `data/nhl_combine/nhl_combine_results.json` is under the repo; load with any JSON reader or use `load_nhl_combine_results()` from the fetch script.
- **MotionApp**: Copy `nhl_combine_results.json` into the app bundle (e.g. Resources) or point to a shared path; parse JSON in Swift.

## Updating data

Run from the Bender repo:

```bash
python scripts/fetch_nhl_combine_data.py
```

This updates `data/nhl_combine/nhl_combine_results.json` from the built-in 2025 top-10 merge. Optionally extend the script to download and parse the NHL PDF for full top-25 or full roster data.
