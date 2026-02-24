"""Paths and settings for March Machine Learning Mania 2026."""
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Competition data (CSVs)
DATA_DIR = PROJECT_ROOT / "datasets"

# Output folder for submissions and artifacts
OUTPUT_DIR = PROJECT_ROOT / "output"

# Target season to predict (Stage 2)
TARGET_SEASON = 2026

# Seasons in Stage 1 submission (2022-2025) — required row count for current l
STAGE1_SEASONS = [2022, 2023, 2024, 2025]
STAGE1_REQUIRED_ROWS = 519_144
# Stage 2 (2026 only) — scores 0.0 until tournaments begin and Kaggle rescores
STAGE2_REQUIRED_ROWS = 132_134
# Seasons used for team strength when predicting 2026 (Stage 2)
RECENT_SEASONS = [2025, 2024, 2023]
