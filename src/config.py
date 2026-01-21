"""Configuration centralisée et constantes du projet."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def get_repo_root(start_path: str | None = None) -> str:
    """Retourne le répertoire racine du repo.

    Objectif: éviter les chemins faux quand le CWD Streamlit n'est pas le repo,
    ou quand le script est lancé depuis un autre dossier.
    """

    def _as_dir(p: Path) -> Path:
        try:
            p = p.resolve()
        except Exception:
            pass
        return p.parent if p.is_file() else p

    def _looks_like_repo_root(p: Path) -> bool:
        return (p / "pyproject.toml").exists() and (p / "src").is_dir()

    starts: list[Path] = []
    if start_path:
        starts.append(_as_dir(Path(start_path)))
    starts.append(_as_dir(Path(__file__)))
    try:
        starts.append(Path.cwd().resolve())
    except Exception:
        starts.append(Path.cwd())

    # Cherche dans les parents proches.
    for s in starts:
        for p in [s] + list(s.parents)[:8]:
            if _looks_like_repo_root(p):
                return str(p)

    # Fallback raisonnable.
    return str(starts[0])


# =============================================================================
# Chemins par défaut
# =============================================================================

def get_default_db_path() -> str:
    """Retourne le chemin par défaut de la DB OpenSpartan Workshop."""
    # Override explicite (utile en Docker/Linux)
    override = os.environ.get("OPENSPARTAN_DB") or os.environ.get("OPENSPARTAN_DB_PATH")
    if override and os.path.exists(override):
        return override

    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return ""
    base = os.path.join(local, "OpenSpartan.Workshop", "data")
    if not os.path.isdir(base):
        return ""
    try:
        dbs = [os.path.join(base, f) for f in os.listdir(base) if f.lower().endswith(".db")]
    except Exception:
        return ""
    if not dbs:
        return ""
    # Si plusieurs DB, prend la plus récente (modification).
    dbs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dbs[0]


def get_default_workshop_exe_path() -> str:
    """Retourne le chemin par défaut de l'exécutable OpenSpartan Workshop."""
    pf86 = os.environ.get("ProgramFiles(x86)")
    if not pf86:
        pf86 = r"C:\Program Files (x86)"
    return os.path.join(pf86, "Den.Dev", "OpenSpartan Workshop", "OpenSpartan.Workshop.exe")


def get_aliases_file_path() -> str:
    """Retourne le chemin du fichier d'alias XUID."""
    override = os.environ.get("OPENSPARTAN_ALIASES_PATH")
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "xuid_aliases.json")


# =============================================================================
# Constantes de l'application
# =============================================================================

# =============================================================================
# Identité par défaut (local)
# =============================================================================

DEFAULT_PLAYER_GAMERTAG = (os.environ.get("OPENSPARTAN_DEFAULT_GAMERTAG") or "").strip()
DEFAULT_PLAYER_XUID = (os.environ.get("OPENSPARTAN_DEFAULT_XUID") or "").strip()


DEFAULT_WAYPOINT_PLAYER = (os.environ.get("OPENSPARTAN_DEFAULT_WAYPOINT_PLAYER") or DEFAULT_PLAYER_GAMERTAG).strip()


# =============================================================================
# Alias XUID par défaut (hardcodés)
# =============================================================================

XUID_ALIASES_DEFAULT: Dict[str, str] = (
    {DEFAULT_PLAYER_XUID: DEFAULT_PLAYER_GAMERTAG}
    if (DEFAULT_PLAYER_XUID and DEFAULT_PLAYER_GAMERTAG)
    else {}
)


# =============================================================================
# Palette de couleurs Halo
# =============================================================================

@dataclass(frozen=True)
class HaloColors:
    """Palette de couleurs inspirée de l'univers Halo."""
    cyan: str = "#35D0FF"
    violet: str = "#8E6CFF"
    green: str = "#3DFFB5"
    red: str = "#FF4D6D"
    amber: str = "#FFB703"
    slate: str = "#A8B2D1"

    def as_dict(self) -> Dict[str, str]:
        """Retourne les couleurs sous forme de dictionnaire."""
        return {
            "cyan": self.cyan,
            "violet": self.violet,
            "green": self.green,
            "red": self.red,
            "amber": self.amber,
            "slate": self.slate,
        }


HALO_COLORS = HaloColors()


# =============================================================================
# Configuration des sessions
# =============================================================================

@dataclass
class SessionConfig:
    """Configuration pour la détection des sessions de jeu."""
    default_gap_minutes: int = 35
    min_gap_minutes: int = 15
    max_gap_minutes: int = 240

    # Seuils pour le bucketing temporel (en jours)
    bucket_threshold_hourly: float = 1.0
    bucket_threshold_daily: float = 6.0
    bucket_threshold_weekly: float = 10.0
    bucket_threshold_monthly: float = 45.0


SESSION_CONFIG = SessionConfig()


# =============================================================================
# Codes de résultat (Outcome)
# =============================================================================

@dataclass(frozen=True)
class OutcomeCodes:
    """Codes de résultat des matchs selon l'API Halo."""
    TIE: int = 1
    WIN: int = 2
    LOSS: int = 3
    NO_FINISH: int = 4

    def to_label(self, code: int) -> str:
        """Convertit un code en label lisible."""
        labels = {
            self.TIE: "Égalité",
            self.WIN: "Victoire",
            self.LOSS: "Défaite",
            self.NO_FINISH: "Non terminé",
        }
        return labels.get(code, "?")


OUTCOME_CODES = OutcomeCodes()


# =============================================================================
# Configuration des graphiques
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration par défaut des graphiques."""
    default_height: int = 360
    tall_height: int = 520
    short_height: int = 320
    
    bar_opacity: float = 0.85
    bar_opacity_secondary: float = 0.65
    line_width: float = 2.2
    
    margin_left: int = 40
    margin_right: int = 20
    margin_top: int = 30
    margin_bottom: int = 40


PLOT_CONFIG = PlotConfig()
