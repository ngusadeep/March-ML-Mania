# March Machine Learning Mania 2026

Forecast the 2026 NCAA Men's and Women's basketball tournaments for [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

## Summary

- **Task:** Predict the probability that the **lower TeamId** beats the **higher TeamId** for every possible matchup in 2026 (men + women).
- **Evaluation:** Brier score (MSE between predicted probabilities and actual 0/1 outcomes).
- **Submission format:** CSV with columns `ID,Pred`. ID = `2026_TeamIdLow_TeamIdHigh`; Pred = P(low beats high). Men's TeamIds 1000–1999, Women's 3000–3999.
- **Data:** `datasets/` — 35 CSVs (teams, seasons, seeds, regular season/tourney results, detailed stats, geography, Massey ordinals, etc.).

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependencies (no pip).

```bash
uv sync
```

Data is expected in `datasets/` (e.g. from the competition or unzipped from `data/march-machine-learning-mania-2026.zip` into `datasets/`).

## Baseline

```bash
uv run python run_baseline.py
```

This builds team strength from 2025/2024/2023 season win% and point differential (regular season + NCAA tourney), then predicts P(low beats high) with a logistic on strength difference. Output: `output/submission.csv`. Upload this to Kaggle as your submission.

## Project layout

- `datasets/` — competition CSVs (M* / W* for men/women, sample submissions Stage1/Stage2).
- `src/config.py` — paths and target season.
- `src/data.py` — load compact results, teams, sample submission; parse ID and gender.
- `src/features.py` — per-team, per-season stats (wins, win%, pts for/against, point diff).
- `src/predict.py` — baseline: strength from recent seasons → probability per matchup.
- `run_baseline.py` — generate `output/submission.csv`.

## Notes

- 2026 submissions will show 0.0 on the leaderboard until the 2026 tournaments are played and Kaggle rescores.
- You must **select two submissions** for scoring; don’t rely on automatic selection.
- Stage 1 sample (`SampleSubmissionStage1.csv`) has 2022–2025 matchups for local validation; Stage 2 has 2026 matchups to predict.
