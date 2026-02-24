#!/usr/bin/env python3
"""
Generate a baseline submission for March Machine Learning Mania 2026.
Uses 2025/2024/2023 season win% and point differential to predict P(lower TeamId wins).
Output: output/submission.csv (ready to upload to Kaggle).
"""
from pathlib import Path

from src.config import OUTPUT_DIR
from src.predict import predict_submission

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sub = predict_submission()
out_path = OUTPUT_DIR / "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Saved {len(sub)} predictions to {out_path}")
