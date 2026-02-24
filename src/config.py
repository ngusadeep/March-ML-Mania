"""Paths and settings for March Machine Learning Mania 2026."""
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Competition data (CSVs)
DATA_DIR = PROJECT_ROOT / "datasets"
# Output folder for submissions and artifacts
OUTPUT_DIR = PROJECT_ROOT / "output"
# Target season to predict
TARGET_SEASON = 2026
# Seasons used for team strength (most recent available before tournament)
RECENT_SEASONS = [2025, 2024, 2023]
