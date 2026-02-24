#!/usr/bin/env python3
"""
Generate a baseline submission for March Machine Learning Mania 2026.
Default: Stage 1 (519,144 rows) for current leaderboard score.
Use --stage2 for 2026-only submission (scores 0.0 until tournaments begin).
"""
import argparse
import sys
from pathlib import Path

from src.config import (
    OUTPUT_DIR,
    STAGE1_REQUIRED_ROWS,
    STAGE1_SEASONS,
    STAGE2_REQUIRED_ROWS,
)
from src.data import load_sample_stage1, load_sample_stage2
from src.features import build_strength_cache
from src.predict import predict_submission


def log(msg: str) -> None:
    print(msg, flush=True)


def validate_submission(sub, required_rows: int, stage_name: str) -> None:
    """Ensure submission has correct row count and format for Kaggle."""
    if len(sub) != required_rows:
        log(f"  ERROR: Expected {required_rows:,} rows, got {len(sub):,}. Fix before submitting.")
        sys.exit(1)
    if list(sub.columns) != ["ID", "Pred"]:
        log("  ERROR: Columns must be exactly ID, Pred.")
        sys.exit(1)
    if not sub["Pred"].between(0.0, 1.0).all():
        log("  ERROR: All Pred values must be between 0 and 1.")
        sys.exit(1)
    log(f"  Valid: {len(sub):,} rows, ID + Pred, Pred in [0,1]. Ready for {stage_name}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate March Madness submission")
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="Generate 2026-only submission (132,134 rows). Scores 0.0 until tournaments begin.",
    )
    args = parser.parse_args()
    use_stage1 = not args.stage2

    log("Started — March Machine Learning Mania 2026 baseline")
    if use_stage1:
        log("Mode: Stage 1 (2022–2025) — submission will get a real leaderboard score.")
    else:
        log("Mode: Stage 2 (2026 only) — submission will score 0.0 until tournaments begin.")
    log("")

    if use_stage1:
        log("Step 1/4 — Loading sample submission (Stage 1: 2022–2025)...")
        sample = load_sample_stage1()
        required_rows = STAGE1_REQUIRED_ROWS
        seasons = STAGE1_SEASONS
    else:
        log("Step 1/4 — Loading sample submission (Stage 2: 2026)...")
        sample = load_sample_stage2()
        required_rows = STAGE2_REQUIRED_ROWS
        seasons = [2025, 2024, 2023]

    log(f"  Loaded {len(sample):,} matchups (required: {required_rows:,}).")
    log("")

    log("Step 2/4 — Building team strength cache (win%, point diff by season)...")
    cache_m = build_strength_cache("M", seasons, verbose=True)
    cache_w = build_strength_cache("W", seasons, verbose=True)
    log(f"  Men: {len(cache_m):,} team-seasons | Women: {len(cache_w):,} team-seasons.")
    log("")

    log("Step 3/4 — Predicting P(lower TeamId wins) for each matchup...")
    sub = predict_submission(
        submission=sample,
        cache_m=cache_m,
        cache_w=cache_w,
        use_stage1=use_stage1,
        verbose=True,
    )
    log("  Done.")
    log("")

    log("Step 4/4 — Validating and writing submission file...")
    validate_submission(
        sub,
        required_rows,
        "current leaderboard" if use_stage1 else "2026 (0.0 until rescore)",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)
    log(f"  Saved {out_path}")
    if not use_stage1:
        log("  → This file will score 0.000 until the 2026 tournaments begin and Kaggle rescores.")
    log("")
    log("Done.")


if __name__ == "__main__":
    main()
    sys.exit(0)
