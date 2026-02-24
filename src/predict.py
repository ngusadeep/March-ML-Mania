"""Baseline: team strength from recent season stats -> P(lower TeamId wins)."""
import numpy as np
import pandas as pd

from .config import RECENT_SEASONS, STAGE1_SEASONS
from .data import gender_from_team_id, load_sample_stage1, load_sample_stage2, parse_submission_id
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
    season: int | None = None,
    seasons_to_try: list[int] | None = None,
    default_pred: float = 0.5,
) -> float:
    """
    Predict P(team_low beats team_high). If season is set, use that season's
    stats first; else try seasons_to_try (default RECENT_SEASONS).
    """
    gender = gender_from_team_id(team_low)
    cache = cache_m if gender == "M" else cache_w
    order = [season] if season is not None else (seasons_to_try or RECENT_SEASONS)
    for s in order:
        sub = cache[cache["Season"] == s]
        low_row = sub[sub["TeamID"] == team_low]
        high_row = sub[sub["TeamID"] == team_high]
        if low_row.empty or high_row.empty:
            continue
        low_row = low_row.iloc[0]
        high_row = high_row.iloc[0]
        s_low = strength_score(low_row)
        s_high = strength_score(high_row)
        logit = s_low - s_high
        pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
        return float(np.clip(pred, 0.01, 0.99))
    return default_pred


def predict_submission(
    submission: pd.DataFrame | None = None,
    cache_m: pd.DataFrame | None = None,
    cache_w: pd.DataFrame | None = None,
    use_stage1: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate Pred for each ID. If use_stage1=True, load Stage 1 (519,144 rows,
    2022-2025) for current leaderboard; else Stage 2 (2026 only).
    """
    if submission is None:
        submission = (
            load_sample_stage1() if use_stage1 else load_sample_stage2()
        )
    seasons = STAGE1_SEASONS if use_stage1 else RECENT_SEASONS
    if cache_m is None:
        cache_m = build_strength_cache("M", seasons)
    if cache_w is None:
        cache_w = build_strength_cache("W", seasons)

    n = len(submission)
    preds = []
    last_pct = -1
    for i, id_str in enumerate(submission["ID"]):
        matchup_season, team_low, team_high = parse_submission_id(id_str)
        pred = predict_one(
            team_low,
            team_high,
            cache_m,
            cache_w,
            season=matchup_season if use_stage1 else None,
            seasons_to_try=RECENT_SEASONS if not use_stage1 else None,
        )
        preds.append(pred)
        if verbose:
            pct = (i + 1) * 100 // n
            if pct >= last_pct + 10 or i + 1 == n:
                print(f"  {pct}% ({i + 1:,} / {n:,})", flush=True)
                last_pct = pct

    out = submission[["ID"]].copy()
    out["Pred"] = preds
    return out
