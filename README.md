# March Machine Learning Mania 2026

Forecast the 2026 NCAA Men's and Women's basketball tournaments for [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

## Summary

- **Task:** Predict the probability that the **lower TeamId** beats the **higher TeamId** for every possible matchup in 2026 (men + women).
- **Evaluation:** Brier score (MSE between predicted probabilities and actual 0/1 outcomes).
- **Submission format:** CSV with columns `ID,Pred`. ID = `2026_TeamIdLow_TeamIdHigh`; Pred = P(low beats high). Men's TeamIds 1000–1999, Women's 3000–3999.
- **Data:** `datasets/` — 35 CSVs (teams, seasons, seeds, regular season/tourney results, detailed stats, geography, Massey ordinals, etc.).

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependencies (no pip).

Data is expected in `datasets/` (e.g. from the competition or unzipped from `data/march-machine-learning-mania-2026.zip` into `datasets/`).

## Steps (with terminal progress)

Run these in order. Example logs show what you should see.

**Step 1 — Sync dependencies**

```bash
uv sync
```

<details>
<summary>Example output</summary>

```
Using Python 3.12.x
Resolved 3 packages in 123ms
Installed 3 packages in 45ms
  + numpy 1.26.x
  + pandas 2.2.x
  + march-ml-mania 0.1.0 (from .)
```

</details>

**Step 2 — Run baseline and generate submission**

```bash
uv run python run_baseline.py
```

<details>
<summary>Example output</summary>

```
Saved 132134 predictions to output/submission.csv
```

</details>

**Step 3 — (Optional) Git: push without large data**

If you use git and keep data in `datasets/`, ensure `datasets/` is in `.gitignore` so large files (e.g. `MMasseyOrdinals.csv` >100 MB) are not committed. To stop tracking them after a mistaken add:

```bash
git rm -r --cached datasets/
git commit -m "Remove large dataset files from tracking"
```

Then rewrite history so the large file never appears in any commit (required for GitHub’s 100 MB limit), then push:

```bash
git checkout --orphan clean-main
git add -A
git commit -m "Initial commit (no datasets)"
git branch -D main
git branch -m main
git push -u origin main --force
```

<details>
<summary>Example output (push)</summary>

```
Enumerating objects: 42, done.
Counting objects: 100% (42/42), done.
...
To https://github.com/youruser/March-ML-Mania.git
 + main -> main (forced update)
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

</details>

---

## Baseline (short)

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
