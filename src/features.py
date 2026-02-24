"""Build team strength features from compact results."""
import pandas as pd

from .data import load_compact_results


def team_season_stats(gender: str, season: int) -> pd.DataFrame:
    """
    For one gender and season, compute per-team: wins, losses, games,
    win_pct, pts_for, pts_against, point_diff_per_game.
    """
    df = load_compact_results(gender)
    df = df[df["Season"] == season]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "TeamID", "Season", "Wins", "Losses", "Games",
                "WinPct", "PtsFor", "PtsAgainst",
                "PtsForPerGame", "PtsAgainstPerGame", "PointDiffPerGame",
            ]
        )

    wins = df.groupby("WTeamID").agg(
        Wins=("WTeamID", "count"),
        PtsFor=("WScore", "sum"),
        PtsAgainst=("LScore", "sum"),
    ).rename_axis("TeamID").reset_index()
    losses = df.groupby("LTeamID").agg(
        Losses=("LTeamID", "count"),
        PtsFor=("LScore", "sum"),
        PtsAgainst=("WScore", "sum"),
    ).rename_axis("TeamID").reset_index()

    teams = wins[["TeamID"]].merge(
        losses[["TeamID"]],
        on="TeamID",
        how="outer",
    ).fillna(0)
    wins_agg = wins.set_index("TeamID")
    losses_agg = losses.set_index("TeamID")

    def get_wins(tid):
        return wins_agg.loc[tid, "Wins"] if tid in wins_agg.index else 0

    def get_losses(tid):
        return losses_agg.loc[tid, "Losses"] if tid in losses_agg.index else 0

    def get_pts_for(tid):
        w = wins_agg.loc[tid, "PtsFor"] if tid in wins_agg.index else 0
        loss_pts = losses_agg.loc[tid, "PtsFor"] if tid in losses_agg.index else 0
        return w + loss_pts

    def get_pts_against(tid):
        w = wins_agg.loc[tid, "PtsAgainst"] if tid in wins_agg.index else 0
        loss_pts = losses_agg.loc[tid, "PtsAgainst"] if tid in losses_agg.index else 0
        return w + loss_pts

    teams["Wins"] = teams["TeamID"].map(get_wins)
    teams["Losses"] = teams["TeamID"].map(get_losses)
    teams["Games"] = teams["Wins"] + teams["Losses"]
    teams["PtsFor"] = teams["TeamID"].map(get_pts_for)
    teams["PtsAgainst"] = teams["TeamID"].map(get_pts_against)
    teams["Season"] = season
    games_safe = teams["Games"].replace(0, 1)
    teams["WinPct"] = teams["Wins"] / games_safe
    teams["PtsForPerGame"] = teams["PtsFor"] / games_safe
    teams["PtsAgainstPerGame"] = teams["PtsAgainst"] / games_safe
    teams["PointDiffPerGame"] = teams["PtsForPerGame"] - teams["PtsAgainstPerGame"]
    return teams


def build_strength_cache(gender: str, seasons: list[int]) -> pd.DataFrame:
    """Stack team-season stats for multiple seasons."""
    parts = [team_season_stats(gender, s) for s in seasons]
    return pd.concat(parts, ignore_index=True)
