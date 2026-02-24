"""March Machine Learning Mania 2026 — forecast NCAA tournament outcomes."""
from .config import DATA_DIR, OUTPUT_DIR, TARGET_SEASON
from .data import load_sample_stage2, load_compact_results, parse_submission_id, gender_from_team_id
from .features import team_season_stats, build_strength_cache
from .predict import predict_submission

__all__ = [
    "DATA_DIR",
    "OUTPUT_DIR",
    "TARGET_SEASON",
    "load_sample_stage2",
    "load_compact_results",
    "parse_submission_id",
    "gender_from_team_id",
    "team_season_stats",
    "build_strength_cache",
    "predict_submission",
]
