"""Récupère un OAuth refresh token via l'auth Azure (SPNKr).

Usage:
- Remplir .env.local (ou .env) avec:
  - SPNKR_AZURE_CLIENT_ID
  - SPNKR_AZURE_CLIENT_SECRET
  - SPNKR_AZURE_REDIRECT_URI (optionnel, défaut https://localhost)

Puis lancer:
  python scripts/spnkr_get_refresh_token.py

Le script affiche le refresh token. Garde-le secret (ne le commit pas).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import re
from typing import Optional
from urllib.parse import parse_qs, quote_plus, urlparse


async def _oauth2_token_request_v2(session, *, app, authorization_code: str) -> str:
    """Échange le `code` contre un refresh token via l'endpoint OAuth2 v2 (consumers).

    Utile quand `login.live.com/oauth20_token.srf` refuse le client_secret (invalid_client)
    alors que l'app est bien une App Registration 'Microsoft identity platform'.
    """

    url = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
    data = {
        "client_id": app.client_id,
        "client_secret": app.client_secret,
        "grant_type": "authorization_code",
        "code": authorization_code,
        "redirect_uri": app.redirect_uri,
        # Scopes Xbox requis.
        "scope": "Xboxlive.signin Xboxlive.offline_access",
    }

    resp = await session.post(url, data=data)
    payload = await resp.json()
    if resp.status >= 400:
        err = payload.get("error")
        desc = payload.get("error_description")
        raise SystemExit(
            "Échec OAuth v2 (consumers).\n"
            f"status={resp.status} error={err}\n"
            f"error_description={desc}"
        )
    refresh = payload.get("refresh_token")
    if not isinstance(refresh, str) or not refresh.strip():
        raise SystemExit("OAuth v2: pas de refresh_token dans la réponse.")
    return refresh.strip()


def _load_dotenv_if_present() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for name in (".env.local", ".env"):
        p = repo_root / name
        if not p.exists():
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and os.environ.get(key) is None:
                os.environ[key] = value


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SPNKr: obtenir un OAuth refresh token (Azure)")
    ap.add_argument(
        "--auth-code",
        default=None,
        help=(
            "Code OAuth (paramètre `code=`) OU URL complète (https://localhost/?code=...). "
            "Alternative: variable d'env SPNKR_OAUTH_AUTH_CODE."
        ),
    )
    ap.add_argument(
        "--auth-url",
        default=None,
        help="Alias de --auth-code (URL complète contenant `code=`).",
    )
    ap.add_argument(
        "--env-file",
        default=None,
        help="Fichier à mettre à jour (défaut: .env.local à la racine du repo).",
    )
    ap.add_argument(
        "--no-write-env-local",
        action="store_true",
        help="N'écrit pas SPNKR_OAUTH_REFRESH_TOKEN dans le fichier env.",
    )
    ap.add_argument(
        "--print-token",
        action="store_true",
        help="Affiche le refresh token dans la sortie (sinon: seulement un message).",
    )
    ap.add_argument(
        "--oauth-endpoint",
        choices=["auto", "live", "v2"],
        default="auto",
        help=(
            "Quel endpoint utiliser pour échanger le code contre un refresh token. "
            "auto: tente SPNKr/login.live.com puis fallback v2 si invalid_client."
        ),
    )
    return ap.parse_args(argv)


def _extract_code(value: str) -> str:
    """Accepte soit un code brut, soit une URL complète contenant `code=`."""
    v = (value or "").strip()
    if not v:
        return ""

    # Cas le plus simple: l'utilisateur colle juste le code.
    if "code=" not in v and "http" not in v:
        return v

    # L'utilisateur colle l'URL complète (souvent https://localhost/?code=...)
    try:
        parsed = urlparse(v)
        qs = parse_qs(parsed.query)
        code_values = qs.get("code")
        if code_values and code_values[0]:
            return str(code_values[0]).strip()
    except Exception:
        pass

    # Fallback: extraction grossière
    if "code=" in v:
        after = v.split("code=", 1)[1]
        return after.split("&", 1)[0].strip()

    return v


def _extract_oauth_error(value: str) -> tuple[Optional[str], Optional[str]]:
    """Extrait (error, error_description) depuis une URL de redirection OAuth."""
    v = (value or "").strip()
    if not v:
        return None, None
    if "error=" not in v and "http" not in v:
        return None, None
    try:
        parsed = urlparse(v)
        qs = parse_qs(parsed.query)
        err = (qs.get("error") or [None])[0]
        desc = (qs.get("error_description") or [None])[0]
        if err:
            return str(err), (str(desc) if desc else None)
    except Exception:
        pass

    # Fallback grossier (au cas où l'URL est mal formée / partielle)
    if "error=" in v:
        err = v.split("error=", 1)[1].split("&", 1)[0].strip() or None
        desc = None
        if "error_description=" in v:
            desc = v.split("error_description=", 1)[1].split("&", 1)[0].strip() or None
        return err, desc
    return None, None


def _build_authorize_url(*, client_id: str, redirect_uri: str) -> str:
    # URL identique à celle imprimée par SPNKr (login.live.com + scopes Xboxlive)
    scope = "Xboxlive.signin Xboxlive.offline_access"
    return (
        "https://login.live.com/oauth20_authorize.srf"
        f"?client_id={quote_plus(client_id)}"
        "&response_type=code"
        "&approval_prompt=auto"
        f"&scope={quote_plus(scope)}"
        f"&redirect_uri={quote_plus(redirect_uri)}"
    )


def _upsert_env_key(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
    else:
        lines = []

    new_lines: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    _load_dotenv_if_present()

    client_id = os.environ.get("SPNKR_AZURE_CLIENT_ID")
    client_secret = os.environ.get("SPNKR_AZURE_CLIENT_SECRET")
    redirect_uri = os.environ.get("SPNKR_AZURE_REDIRECT_URI") or "https://localhost"

    if not client_id or not client_secret:
        raise SystemExit(
            "Variables Azure manquantes. Renseigne SPNKR_AZURE_CLIENT_ID et SPNKR_AZURE_CLIENT_SECRET dans .env.local/.env."
        )

    # Diagnostic (sans afficher de secret): utile quand Azure renvoie unauthorized_client.
    secret_len = len(client_secret)
    is_uuidish = bool(re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", client_secret.strip()))
    print("\n[Config Azure]")
    print(f"- SPNKR_AZURE_CLIENT_ID: {client_id}")
    print(f"- SPNKR_AZURE_REDIRECT_URI: {redirect_uri}")
    print(f"- SPNKR_AZURE_CLIENT_SECRET: set (len={secret_len})")
    if is_uuidish:
        print(
            "Attention: SPNKR_AZURE_CLIENT_SECRET ressemble à un UUID. "
            "Tu as peut-être copié le 'Secret ID' au lieu de la 'Value'."
        )

    try:
        from aiohttp import ClientSession
        from spnkr import AzureApp, authenticate_player
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Dépendance manquante. Installe SPNKr d'abord, ex:\n"
            "pip install 'spnkr @ git+https://github.com/acurtis166/SPNKr.git'"
        ) from e

    app = AzureApp(client_id, client_secret, redirect_uri)

    # Mode non-interactif: si pas de code fourni, on affiche juste l'URL d'autorisation et on sort.
    provided = args.auth_code or args.auth_url or os.environ.get("SPNKR_OAUTH_AUTH_CODE")
    if not provided:
        authorize_url = _build_authorize_url(client_id=client_id, redirect_uri=redirect_uri)
        print(
            "\nÉtapes (non-interactif):\n"
            "1) Ouvre l'URL ci-dessous et connecte-toi.\n"
            "2) À la fin, tu seras redirigé vers https://localhost (souvent une page d'erreur).\n"
            "   C'est normal: copie simplement l'URL de la barre d'adresse (elle contient `code=`).\n"
            "3) Relance le script *tout de suite* (le `code` expire vite et est à usage unique) avec:\n"
            '   python scripts/spnkr_get_refresh_token.py --oauth-endpoint v2 --auth-code "https://localhost/?code=..."\n'
        )
        print(authorize_url)
        return 0

    # Si l'utilisateur colle une URL https://localhost/?error=..., on explique quoi corriger.
    err, desc = _extract_oauth_error(str(provided))
    if err:
        msg = [
            "L'auth OAuth a renvoyé une erreur (pas de `code=` dans l'URL).",
            f"error={err}",
        ]
        if desc:
            msg.append(f"error_description={desc}")

        msg.append("\nCauses fréquentes:")
        msg.append(
            "- `unauthorized_client` / mention de secret: tu as copié le mauvais champ, ou tu n'as pas de Client secret actif. "
            "Dans Azure: Certificates & secrets → New client secret → copie la *Value* (pas le Secret ID)."
        )
        if desc and "consumers" in desc.lower():
            msg.append(
                "- 'not enabled for consumers': ton App Registration n'accepte pas les comptes Microsoft personnels. "
                "Dans Azure: App registrations → (ton app) → Authentication/Branding (selon UI) → Supported account types: "
                "choisis une option incluant **personal Microsoft accounts**. "
                "Alternative: Manifest → `signInAudience` = `AzureADandPersonalMicrosoftAccount` (ou `PersonalMicrosoftAccount`)."
            )
        msg.append(
            "- pas de `personal Microsoft accounts`: à la création de l'app, l'option doit inclure les comptes Microsoft personnels."
        )
        msg.append(
            "- redirect mismatch: le Redirect URI (type Web) doit être exactement `https://localhost` et correspondre à SPNKR_AZURE_REDIRECT_URI."
        )

        raise SystemExit("\n".join(msg))

    auth_code = _extract_code(str(provided))
    if not auth_code:
        raise SystemExit("Impossible d'extraire `code=` depuis la valeur fournie.")

    import builtins

    original_input = builtins.input

    def _fake_input(prompt: str = "") -> str:  # type: ignore[override]
        return auth_code

    builtins.input = _fake_input

    async with ClientSession() as session:
        try:
            if args.oauth_endpoint == "v2":
                refresh_token = await _oauth2_token_request_v2(session, app=app, authorization_code=auth_code)
            else:
                refresh_token = await authenticate_player(session, app)
        except Exception as e:
            handled = False
            # Message ultra clair pour les erreurs OAuth courantes.
            try:
                from spnkr.errors import OAuth2Error

                if isinstance(e, OAuth2Error):
                    raw = getattr(e, "raw", None)
                    if isinstance(raw, dict):
                        err2 = raw.get("error")
                        desc2 = raw.get("error_description")
                        if err2 == "invalid_client" and isinstance(desc2, str) and "client_secret" in desc2:
                            if args.oauth_endpoint == "auto":
                                # Tentative de fallback: endpoint v2 (consumers)
                                refresh_token = await _oauth2_token_request_v2(session, app=app, authorization_code=auth_code)
                                print(
                                    "\nInfo: `login.live.com` a refusé le client_secret (invalid_client). "
                                    "Fallback réussi via endpoint OAuth v2 (consumers)."
                                )
                                handled = True
                            else:
                                raise SystemExit(
                                    "Azure a refusé le client: `invalid_client` (client_secret invalide).\n\n"
                                    "Ça arrive quand:\n"
                                    "- le `SPNKR_AZURE_CLIENT_SECRET` ne correspond pas à ce `SPNKR_AZURE_CLIENT_ID` (tu as 2 apps, secret sur l'autre),\n"
                                    "- tu as copié un champ incorrect,\n"
                                    "- le secret a expiré.\n\n"
                                    f"Client ID actuel: {client_id}\n"
                                    "Fix: ouvre cette App Registration (même client id) → Certificates & secrets → New client secret,\n"
                                    "copie la **Value** et remplace `SPNKR_AZURE_CLIENT_SECRET` dans .env.local.\n"
                                    "Ensuite regénère un nouveau `code=` (un code OAuth est à usage unique et peut expirer rapidement)."
                                ) from e
            except ModuleNotFoundError:
                pass

            if not handled:
                raise
        finally:
            try:
                builtins.input = original_input
            except Exception:
                pass

    # Si on est passé par un fallback, refresh_token est déjà défini.

    env_path = Path(args.env_file) if args.env_file else (Path(__file__).resolve().parent.parent / ".env.local")
    if not args.no_write_env_local:
        _upsert_env_key(env_path, "SPNKR_OAUTH_REFRESH_TOKEN", refresh_token)
        print(f"\nOK: refresh token écrit dans {env_path}")
    else:
        print("\nOK: refresh token récupéré (non écrit dans un fichier).")

    if args.print_token or args.no_write_env_local:
        print("\n=== SPNKr OAuth refresh token ===\n")
        print(refresh_token)

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(asyncio.run(main(sys.argv[1:])))
