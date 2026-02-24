#!/usr/bin/env python3
"""
Generate a baseline submission for March Machine Learning Mania 2026.
Uses 2025/2024/2023 season win% and point differential to predict P(lower TeamId wins).
Output: output/submission.csv (ready to upload to Kaggle).
"""
import sys
from pathlib import Path

from src.config import OUTPUT_DIR, RECENT_SEASONS
from src.data import load_sample_stage2
from src.features import build_strength_cache
from src.predict import predict_submission


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    log("Started — March Machine Learning Mania 2026 baseline")
    log("")

    log("Step 1/4 — Loading sample submission (Stage 2)...")
    sample = load_sample_stage2()
    log(f"  Loaded {len(sample):,} matchups.")
    log("")

    log("Step 2/4 — Building team strength cache (win%, point diff by season)...")
    cache_m = build_strength_cache("M", RECENT_SEASONS, verbose=True)
    cache_w = build_strength_cache("W", RECENT_SEASONS, verbose=True)
    log(f"  Men: {len(cache_m):,} team-seasons | Women: {len(cache_w):,} team-seasons.")
    log("")

    log("Step 3/4 — Predicting P(lower TeamId wins) for each matchup...")
    sub = predict_submission(
        submission=sample, cache_m=cache_m, cache_w=cache_w, verbose=True
    )
    log("  Done.")
    log("")

    log("Step 4/4 — Writing submission file...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)
    log(f"  Saved {len(sub):,} predictions to {out_path}")
    log("")
    log("Done.")


if __name__ == "__main__":
    main()
    sys.exit(0)
