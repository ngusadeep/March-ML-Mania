"""Baseline: team strength from recent season stats -> P(lower TeamId wins)."""
import numpy as np
import pandas as pd

from .config import RECENT_SEASONS
from .data import gender_from_team_id, load_sample_stage2, parse_submission_id
from .features import build_strength_cache


def strength_score(row: pd.Series) -> float:
    """Single number strength from WinPct and point diff (higher = better)."""
    win_pct = row.get("WinPct", 0.5)
    pt_diff = row.get("PointDiffPerGame", 0.0)
    return win_pct + 0.002 * pt_diff


def predict_one(
    team_low: int,
    team_high: int,
    cache_m: pd.DataFrame,
    cache_w: pd.DataFrame,
    default_pred: float = 0.5,
) -> float:
    """
    Predict P(team_low beats team_high) using most recent season stats.
    Uses 2025 if available, else 2024, etc. Fallback default_pred if team missing.
    """
    gender = gender_from_team_id(team_low)
    cache = cache_m if gender == "M" else cache_w
    # Use most recent season available for each team
    for season in RECENT_SEASONS:
        sub = cache[cache["Season"] == season]
        low_row = sub[sub["TeamID"] == team_low]
        high_row = sub[sub["TeamID"] == team_high]
        if low_row.empty or high_row.empty:
            continue
        low_row = low_row.iloc[0]
        high_row = high_row.iloc[0]
        s_low = strength_score(low_row)
        s_high = strength_score(high_row)
        # P(low wins) via logistic: 1 / (1 + exp(-(s_low - s_high)))
        logit = s_low - s_high
        pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
        return float(np.clip(pred, 0.01, 0.99))
    return default_pred


def predict_submission(
    submission: pd.DataFrame | None = None,
    cache_m: pd.DataFrame | None = None,
    cache_w: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate Pred for each ID in the Stage 2 sample (or provided submission).
    """
    if submission is None:
        submission = load_sample_stage2()
    if cache_m is None:
        cache_m = build_strength_cache("M", RECENT_SEASONS)
    if cache_w is None:
        cache_w = build_strength_cache("W", RECENT_SEASONS)

    preds = []
    for id_str in submission["ID"]:
        _, team_low, team_high = parse_submission_id(id_str)
        pred = predict_one(team_low, team_high, cache_m, cache_w)
        preds.append(pred)

    out = submission[["ID"]].copy()
    out["Pred"] = preds
    return out
