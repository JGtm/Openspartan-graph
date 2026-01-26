"""Construction d'URLs pour les assets Halo Waypoint.

Convertit les chemins d'inventaire en URLs d'images téléchargeables.
"""

from __future__ import annotations


def _to_image_url(path: str | None) -> str | None:
    """Convertit un chemin d'asset en URL d'image complète.
    
    Args:
        path: Chemin relatif ou URL de l'image.
        
    Returns:
        URL complète vers l'image, ou None si le chemin est vide.
    """
    p = str(path or "").strip()
    if not p:
        return None
    if p.startswith("http://") or p.startswith("https://"):
        return p

    host = "https://gamecms-hacs.svc.halowaypoint.com"

    # Certains champs retournent déjà un chemin vers /hi/images/file/...
    p_lower = p.lower()
    marker = "/hi/images/file/"
    if marker in p_lower:
        rel = p[p_lower.index(marker) :]
        # Normalise la casse comme observé (hi/Images/file)
        rel = rel.replace("/hi/images/file/", "/hi/Images/file/")
        return f"{host}{rel}"

    rel = p.lstrip("/")
    return f"{host}/hi/Images/file/{rel}"


def _inventory_emblem_to_waypoint_png(emblem_path: str | None, configuration_id: int | None) -> str | None:
    """Best-effort: convertit un chemin Inventory/Spartan/Emblems/<name>.json vers une image PNG Waypoint.

    Pattern observé:
    - Inventory/Spartan/Emblems/<stem>.json + configuration_id ->
      /hi/Waypoint/file/images/emblems/<stem>_<configuration_id>.png
    
    Note: Ce pattern ne fonctionne pas pour tous les emblèmes (ex: ranked, certains events).
    Dans ces cas, il faut appeler l'API progression pour récupérer le PNG via DisplayPath.
    """
    p = str(emblem_path or "").strip()
    if not p or configuration_id is None:
        return None
    try:
        cfg = int(configuration_id)
    except Exception:
        return None
    if cfg <= 0:
        return None

    if "/Spartan/Emblems/" not in p:
        return None

    # Récupère le stem sans extension (.json)
    name = p.split("/Spartan/Emblems/", 1)[1].split("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    if not stem:
        return None

    host = "https://gamecms-hacs.svc.halowaypoint.com"
    return f"{host}/hi/Waypoint/file/images/emblems/{stem}_{cfg}.png"


def _inventory_json_to_cms_url(inventory_path: str | None) -> str | None:
    """Construit l'URL vers le fichier JSON CMS pour un chemin d'inventaire.
    
    Ex: Inventory/Spartan/Emblems/104-001-olympus-stuck-3d208338.json
      -> https://gamecms-hacs.svc.halowaypoint.com/hi/progression/file/Inventory/Spartan/Emblems/104-001-olympus-stuck-3d208338.json
    """
    p = str(inventory_path or "").strip()
    if not p:
        return None
    
    # Normaliser le chemin
    if p.startswith("/"):
        p = p[1:]
    
    # Vérifier que c'est un chemin d'inventaire
    p_lower = p.lower()
    if not (p_lower.startswith("inventory/") or "/inventory/" in p_lower):
        return None
    
    host = "https://gamecms-hacs.svc.halowaypoint.com"
    return f"{host}/hi/progression/file/{p}"


def _waypoint_nameplate_png_from_emblem(emblem_path: str | None, configuration_id: int | None) -> str | None:
    """Best-effort: construit une URL nameplate Waypoint basée sur l'emblem.

    Pattern observé:
    - /hi/Waypoint/file/images/nameplates/<emblem_stem>_<configuration_id>.png
    
    Note: Le configuration_id est un entier 32 bits signé dans l'API.
    Nous utilisons la valeur signée directement car c'est ce que Waypoint attend.
    Certaines combinaisons emblem + configuration_id n'ont pas de nameplate générée.
    """
    p = str(emblem_path or "").strip()
    if not p or configuration_id is None:
        return None
    try:
        cfg = int(configuration_id)
    except Exception:
        return None
    # Accepter tous les configuration_id non-nuls (positifs ou négatifs)
    if cfg == 0:
        return None

    if "/Spartan/Emblems/" not in p:
        return None

    name = p.split("/Spartan/Emblems/", 1)[1].split("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    if not stem:
        return None

    host = "https://gamecms-hacs.svc.halowaypoint.com"
    return f"{host}/hi/Waypoint/file/images/nameplates/{stem}_{cfg}.png"


def _inventory_backdrop_to_waypoint_png(backdrop_path: str | None) -> str | None:
    """Best-effort: convertit un chemin Inventory/Spartan/BackdropImages/<name>.json vers une image PNG Waypoint.

    Pattern observé:
    - Inventory/Spartan/BackdropImages/<stem>.json ->
      /hi/Waypoint/file/images/backdrops/<stem>.png
    
    ATTENTION: Ce pattern ne fonctionne PAS pour la majorité des backdrops !
    Les backdrops utilisent généralement des chemins comme `progression/backgrounds/...`
    qui ne sont pas dans le mapping Waypoint.
    
    Cette fonction ne retourne une URL que si le backdrop est déjà un chemin PNG direct.
    Pour les autres cas, utiliser _resolve_inventory_png_via_api().
    """
    p = str(backdrop_path or "").strip()
    if not p:
        return None

    # Si c'est déjà un PNG/une vraie image, retourner tel quel
    p_lower = p.lower()
    if p_lower.endswith(".png") or p_lower.endswith(".jpg") or p_lower.endswith(".jpeg"):
        return _to_image_url(p)

    # Pour les fichiers JSON, ne pas utiliser le pattern Waypoint (ne fonctionne pas)
    # -> Laisser le caller utiliser _resolve_inventory_png_via_api() à la place
    return None


async def resolve_inventory_png_via_api(
    session,
    inventory_path: str | None,
    *,
    spartan_token: str,
    clearance_token: str,
) -> str | None:
    """Résout un chemin Inventory/*.json vers l'URL du PNG réel via l'API progression.
    
    Endpoint: GET https://gamecms-hacs.svc.halowaypoint.com/hi/progression/file/{inventory_path}
    
    Cette API retourne un JSON avec la structure :
    ```json
    {
      "CommonData": {
        "DisplayPath": {
          "Media": {
            "MediaUrl": {
              "Path": "progression/Backdrops/103-000-ui-background.png"
            }
          }
        }
      }
    }
    ```
    
    L'URL finale du PNG est : /hi/images/file/{Path}
    
    Retourne l'URL complète vers le PNG, ou None si non résolu.
    """
    cms_url = _inventory_json_to_cms_url(inventory_path)
    if not cms_url:
        return None
    
    headers = {
        "Accept": "application/json",
        "X-343-Authorization-Spartan": spartan_token,
        "343-Clearance": clearance_token,
    }
    
    try:
        async with session.get(cms_url, headers=headers) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None
    
    # Extraire CommonData.DisplayPath.Media.MediaUrl.Path
    common_data = data.get("CommonData", {})
    display_path = common_data.get("DisplayPath", {})
    media = display_path.get("Media", {})
    media_url = media.get("MediaUrl", {})
    png_path = media_url.get("Path", "")
    
    if not png_path:
        # Fallback 1: essayer FolderPath + FileName dans DisplayPath
        folder = display_path.get("FolderPath", "")
        filename = display_path.get("FileName", "")
        if folder and filename:
            png_path = f"{folder}/{filename}"
    
    if not png_path:
        # Fallback 2: essayer ImagePath.Media.MediaUrl.Path (structure Grunt/Backdrop)
        image_path = data.get("ImagePath", {})
        img_media = image_path.get("Media", {})
        img_media_url = img_media.get("MediaUrl", {})
        png_path = img_media_url.get("Path", "")
    
    if not png_path:
        return None
    
    # Construire l'URL complète
    png_path = png_path.lstrip("/")
    host = "https://gamecms-hacs.svc.halowaypoint.com"
    return f"{host}/hi/images/file/{png_path}"
