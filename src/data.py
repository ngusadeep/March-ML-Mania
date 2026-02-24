"""Load competition data from datasets/."""
import pandas as pd

from .config import DATA_DIR


def load_compact_results(gender: str) -> pd.DataFrame:
    """Load regular season + NCAA tourney compact results for M or W."""
    reg = pd.read_csv(DATA_DIR / f"{gender}RegularSeasonCompactResults.csv")
    tourney = pd.read_csv(DATA_DIR / f"{gender}NCAATourneyCompactResults.csv")
    return pd.concat([reg, tourney], ignore_index=True)


def load_teams(gender: str) -> pd.DataFrame:
    """Load team list for M or W."""
    return pd.read_csv(DATA_DIR / f"{gender}Teams.csv")


def load_sample_stage2() -> pd.DataFrame:
    """Load Stage 2 sample submission (2026 matchups to predict)."""
    return pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")


def load_sample_stage1() -> pd.DataFrame:
    """Load Stage 1 sample (2022–2025 matchups, for validation)."""
    return pd.read_csv(DATA_DIR / "SampleSubmissionStage1.csv")


def parse_submission_id(id_str: str) -> tuple[int, int, int]:
    """Parse 'SSSS_XXXX_YYYY' -> (season, team_low, team_high)."""
    season_str, low_str, high_str = id_str.split("_")
    return int(season_str), int(low_str), int(high_str)


def gender_from_team_id(team_id: int) -> str:
    """Men 1000–1999 -> 'M', Women 3000–3999 -> 'W'."""
    return "M" if 1000 <= team_id < 2000 else "W"
