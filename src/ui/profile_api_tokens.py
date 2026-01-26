"""Gestion des tokens d'authentification SPNKr/Halo Waypoint.

Chargement des variables d'environnement et refresh des tokens via Azure.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from pathlib import Path


def _repo_root() -> Path:
    """Retourne la racine du repository."""
    return Path(__file__).resolve().parents[2]


def _load_dotenv_if_present() -> None:
    """Charge `.env.local` puis `.env` à la racine du repo (si présents).

    Objectif: permettre à Streamlit (et aux helpers UI) de trouver les variables
    SPNKr Azure sans exiger une export manuelle dans le shell.
    Règles:
    - lignes `KEY=VALUE`
    - ignore lignes vides / commentaires (#)
    - ne remplace pas une variable déjà définie dans l'environnement
    """
    repo_root = _repo_root()
    for name in (".env.local", ".env"):
        dotenv_path = repo_root / name
        if not dotenv_path.exists():
            continue
        try:
            content = dotenv_path.read_text(encoding="utf-8")
        except Exception:
            continue

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if os.environ.get(key) is None:
                os.environ[key] = value


def _is_probable_auth_error(err: Exception) -> bool:
    """Détecte si une erreur est probablement due à un problème d'authentification."""
    msg = str(err or "")
    m = msg.lower()
    # Heuristique: SPNKr/aiohttp remontent souvent 401/unauthorized dans le message.
    return ("401" in m) or ("unauthorized" in m) or ("forbidden" in m and "403" in m)


async def get_tokens(
    session,
    *,
    spartan_token: str | None,
    clearance_token: str | None,
    timeout_seconds: int,
) -> tuple[str, str]:
    """Récupère ou rafraîchit les tokens SPNKr.
    
    Args:
        session: Session aiohttp.
        spartan_token: Token Spartan existant (ou None pour en obtenir un nouveau).
        clearance_token: Token Clearance existant (ou None pour en obtenir un nouveau).
        timeout_seconds: Timeout pour les appels réseau.
        
    Returns:
        Tuple (spartan_token, clearance_token).
        
    Raises:
        RuntimeError: Si les tokens ne peuvent pas être obtenus.
    """
    _load_dotenv_if_present()
    st = str(spartan_token or "").strip()
    ct = str(clearance_token or "").strip()
    if st and ct:
        # Rend les tokens accessibles aux autres helpers (ex: téléchargement d'assets).
        os.environ["SPNKR_SPARTAN_TOKEN"] = st
        os.environ["SPNKR_CLEARANCE_TOKEN"] = ct
        return st, ct

    # Fallback: récupère via Azure refresh token (opt-in)
    azure_client_id = str(os.environ.get("SPNKR_AZURE_CLIENT_ID") or "").strip()
    azure_client_secret = str(os.environ.get("SPNKR_AZURE_CLIENT_SECRET") or "").strip()
    azure_redirect_uri = str(os.environ.get("SPNKR_AZURE_REDIRECT_URI") or "").strip() or "https://localhost"
    oauth_refresh_token = str(os.environ.get("SPNKR_OAUTH_REFRESH_TOKEN") or "").strip()

    if not (azure_client_id and azure_client_secret and oauth_refresh_token):
        raise RuntimeError(
            "Tokens manquants: définis soit SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN, "
            "soit SPNKR_AZURE_CLIENT_ID + SPNKR_AZURE_CLIENT_SECRET + SPNKR_OAUTH_REFRESH_TOKEN."
        )

    from spnkr import AzureApp, refresh_player_tokens

    async def _refresh_oauth_access_token_v2(refresh_token: str, app: AzureApp) -> str:
        url = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
        data = {
            "client_id": app.client_id,
            "client_secret": app.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": "Xboxlive.signin Xboxlive.offline_access",
        }
        resp = await session.post(url, data=data)
        payload = await resp.json()
        if resp.status >= 400:
            raise RuntimeError(
                "Échec refresh OAuth v2 (consumers). "
                f"status={resp.status} error={payload.get('error')} desc={payload.get('error_description')}"
            )
        access = payload.get("access_token")
        if not isinstance(access, str) or not access.strip():
            raise RuntimeError("OAuth v2: pas de access_token dans la réponse.")
        return access.strip()

    app = AzureApp(azure_client_id, azure_client_secret, azure_redirect_uri)
    try:
        player = await refresh_player_tokens(session, app, oauth_refresh_token)
        st2, ct2 = str(player.spartan_token.token), str(player.clearance_token.token)
        os.environ["SPNKR_SPARTAN_TOKEN"] = st2
        os.environ["SPNKR_CLEARANCE_TOKEN"] = ct2
        return st2, ct2
    except Exception as e:
        msg = str(e)
        if "invalid_client" not in msg or "client_secret" not in msg:
            raise

        # Fallback: endpoint OAuth v2 (consumers) -> chain Xbox/XSTS/Halo
        from spnkr.auth.xbox import request_user_token, request_xsts_token
        from spnkr.auth.core import XSTS_V3_HALO_AUDIENCE, XSTS_V3_XBOX_AUDIENCE
        from spnkr.auth.halo import request_spartan_token, request_clearance_token

        access_token = await _refresh_oauth_access_token_v2(oauth_refresh_token, app)
        user_token = await request_user_token(session, access_token)
        _ = await request_xsts_token(session, user_token.token, XSTS_V3_XBOX_AUDIENCE)
        halo_xsts_token = await request_xsts_token(session, user_token.token, XSTS_V3_HALO_AUDIENCE)
        spartan = await request_spartan_token(session, halo_xsts_token.token)
        clearance = await request_clearance_token(session, spartan.token)
        st3, ct3 = str(spartan.token), str(clearance.token)
        os.environ["SPNKR_SPARTAN_TOKEN"] = st3
        os.environ["SPNKR_CLEARANCE_TOKEN"] = ct3
        return st3, ct3


def ensure_spnkr_tokens(*, timeout_seconds: int = 12) -> tuple[bool, str | None]:
    """Best-effort: s'assure que SPNKR_SPARTAN_TOKEN + SPNKR_CLEARANCE_TOKEN sont disponibles.

    Utile quand on utilise le cache d'apparence (donc pas d'appel API au run courant),
    mais qu'on veut quand même télécharger des assets /hi/Images/file/ qui exigent une auth.

    Returns:
        Tuple (ok, error_message).
    """

    def _run_sync(coro):
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            msg = str(e)
            if "asyncio.run() cannot be called" not in msg:
                raise
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: asyncio.run(coro))
                return fut.result(timeout=float(timeout_seconds) + 20.0)

    async def _run() -> None:
        import aiohttp

        st = str(os.environ.get("SPNKR_SPARTAN_TOKEN") or "").strip() or None
        ct = str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip() or None

        timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await get_tokens(
                session,
                spartan_token=st,
                clearance_token=ct,
                timeout_seconds=int(timeout_seconds),
            )

    try:
        _run_sync(_run())
        st = str(os.environ.get("SPNKR_SPARTAN_TOKEN") or "").strip()
        ct = str(os.environ.get("SPNKR_CLEARANCE_TOKEN") or "").strip()
        if st and ct:
            return True, None
        return False, "Tokens SPNKr introuvables (env et Azure refresh non configurés)."
    except Exception as e:
        return False, str(e)
