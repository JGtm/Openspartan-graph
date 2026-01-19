"""Gestion des connexions SQLite."""

import os
import sqlite3
import urllib.parse
from contextlib import contextmanager
from typing import Generator


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    v = value.strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _connect_sqlite(db_path: str) -> sqlite3.Connection:
    """Ouvre une connexion SQLite.

    En Docker, la DB peut être montée en lecture seule (volume `:ro`).
    Dans ce cas, on retente automatiquement en mode read-only.
    """
    def ro_uri(path: str) -> str:
        abs_path = os.path.abspath(path)
        if os.name == "nt":
            abs_path = abs_path.replace("\\", "/")
            if len(abs_path) >= 2 and abs_path[1] == ":":
                abs_path = "/" + abs_path
            encoded = urllib.parse.quote(abs_path, safe="/:")
            return f"file:{encoded}?mode=ro"
        encoded = urllib.parse.quote(abs_path, safe="/")
        return f"file:{encoded}?mode=ro"

    force_ro = _is_truthy(os.environ.get("OPENSPARTAN_DB_READONLY"))
    if force_ro:
        return sqlite3.connect(ro_uri(db_path), uri=True)

    try:
        return sqlite3.connect(db_path)
    except sqlite3.OperationalError:
        # Fallback read-only si le fichier existe mais n'est pas inscriptible.
        if os.path.exists(db_path):
            return sqlite3.connect(ro_uri(db_path), uri=True)
        raise


class DatabaseConnection:
    """Gestionnaire de connexion SQLite avec context manager.
    
    Exemple d'utilisation:
        with DatabaseConnection("path/to/db.db") as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM table")
    """

    def __init__(self, db_path: str):
        """Initialise la connexion.
        
        Args:
            db_path: Chemin vers le fichier SQLite.
        """
        self.db_path = db_path
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        """Ouvre la connexion."""
        self._connection = _connect_sqlite(self.db_path)
        return self._connection

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ferme la connexion."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None


@contextmanager
def get_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager pour obtenir une connexion SQLite.
    
    Args:
        db_path: Chemin vers le fichier SQLite.
        
    Yields:
        La connexion SQLite ouverte.
        
    Exemple:
        with get_connection("path/to/db.db") as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM table")
    """
    con = _connect_sqlite(db_path)
    try:
        yield con
    finally:
        con.close()
