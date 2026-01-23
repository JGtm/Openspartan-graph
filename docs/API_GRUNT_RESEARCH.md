# Recherche API Grunt / Halo Infinite - Référence Future

> **Date de recherche** : 22 janvier 2026  
> **Contexte** : OpenSpartan Workshop utilise la bibliothèque Grunt (Den.Dev.Grunt) qui n'est pas publique.  
> **Objectif** : Documenter les endpoints et structures pour implémenter ces fonctionnalités quand l'API sera disponible.

---

## 📋 Résumé des fonctionnalités et stabilité

| Fonctionnalité | Endpoint | SPNKr | Stabilité |
|----------------|----------|-------|----------|
| **Service Tag** | `economy.get_player_customization` | ✅ | 🟢 **Stable** |
| **Emblem path** | `economy.get_player_customization` | ✅ | 🟢 **Stable** |
| **Backdrop path** | `economy.get_player_customization` | ✅ | 🟢 **Stable** |
| **Emblem PNG URL** | Pattern Waypoint + Progression API | ⚠️ Workaround | 🟡 Partiel |
| **Backdrop PNG URL** | Progression API | ⚠️ Workaround | 🟡 Partiel |
| **Career Rank metadata** | `gamecms_hacs.get_career_reward_track()` | ✅ | 🟢 **Stable** |
| **Career Rank progression** | GET `/hi/players/xuid({XUID})/rewardtracks/careerranks/careerrank1` | ⚠️ Custom | 🟢 **Stable** (den.dev) |
| **CSR (Competitive Skill)** | `skill.get_playlist_csr` | ✅ | 🟢 **Stable** |

---

## � Endpoints confirmés STABLES

### 1. Player Customization (Service Tag, Emblem, Backdrop)

**Endpoint** (SPNKr `economy.get_player_customization`):
```
GET https://economy.svc.halowaypoint.com:443/hi/players/xuid({XUID})/customization?view=public
```

**Headers requis** :
- `x-343-authorization-spartan: {SPARTAN_TOKEN}`
- `343-clearance: {CLEARANCE_TOKEN}`

**Réponse** (extrait pertinent):
```json
{
  "Appearance": {
    "ServiceTag": "JTGM",
    "BackdropImagePath": "Inventory/Spartan/BackdropImages/103-000-ui-background-e86f6dee.json",
    "Emblem": {
      "EmblemPath": "Inventory/Spartan/Emblems/104-001-olympus-stuck-3d208338.json",
      "ConfigurationId": 12345678
    }
  }
}
```

**Fiabilité** : 🟢 **Très stable** - C'est l'API officielle utilisée par Halo Waypoint.

---

### 2. Career Rank - Progression du joueur

**Endpoint confirmé** (source: [den.dev/blog/halo-infinite-career-api](https://den.dev/blog/halo-infinite-career-api/)) :
```
GET https://economy.svc.halowaypoint.com/hi/players/xuid({XUID})/rewardtracks/careerranks/careerrank1
```

> ⚠️ **Note** : L'URL utilise `careerranks/careerrank1` (avec un 's' et en minuscules)

**Headers requis** :
- `x-343-authorization-spartan: {SPARTAN_TOKEN}`
- `343-clearance: {CLEARANCE_TOKEN}`

**Réponse attendue** (format den.dev - confirmé) :
```json
{
  "RewardTrackPath": "RewardTracks/CareerRanks/careerRank1.json",
  "TrackType": "CareerRank",
  "CurrentProgress": {
    "Rank": 167,
    "PartialProgress": 8050,
    "IsOwned": false,
    "HasReachedMaxRank": false
  },
  "PreviousProgress": null,
  "IsOwned": false,
  "BaseXp": null,
  "BoostXp": null
}
```

**Réponse alternative** (format Grunt/batch - POST) :
```json
{
  "RewardTracks": [
    {
      "Result": {
        "CurrentProgress": {
          "Rank": 42,
          "PartialProgress": 1250
        }
      }
    }
  ]
}
```

**Code Python implémenté** :
```python
# Format 1 (den.dev) - GET direct par joueur
career_url = f"https://economy.svc.halowaypoint.com/hi/players/xuid({xuid})/rewardtracks/careerranks/careerrank1"
async with session.get(career_url, headers=headers) as resp:
    if resp.status == 200:
        data = await resp.json()
        current_progress = data.get("CurrentProgress", {})
        rank = current_progress.get("Rank")
        partial_xp = current_progress.get("PartialProgress", 0)
```

**Logique d'affichage du rang** :
```python
# Note: Rank 272 = Hero (rang max), sinon rank+1 pour l'affichage
display_rank = current_rank if current_rank == 272 else current_rank + 1
```

---

### 2. Career Rank - Métadonnées des rangs

**Endpoint** (disponible dans SPNKr) :
```
GET https://gamecms-hacs.svc.halowaypoint.com/hi/Progression/file/RewardTracks/CareerRanks/careerRank1.json
```

**Structure SPNKr disponible** :
```python
from spnkr.models.gamecms_hacs import CareerRewardTrack, CareerRewardTrackRank

# CareerRewardTrackRank contient :
# - rank: int                    # Numéro du rang (0-272)
# - rank_title: str              # "Private", "Sergeant", "General", "Hero"
# - rank_sub_title: str          # "Bronze", "Silver", "Gold", "Onyx"
# - rank_tier: str               # "1", "2", "3"
# - tier_type: str               # "Bronze", "Silver", etc.
# - rank_icon: str               # Chemin icône 280px
# - rank_large_icon: str         # Chemin icône 600px (célébration)
# - rank_adornment_icon: str     # Chemin icône adornment 196px
# - xp_required_for_rank: int    # XP requis
```

**Exemple d'utilisation** :
```python
async def get_career_rank_metadata(client):
    resp = await client.gamecms_hacs.get_career_reward_track()
    career_track = resp.data
    
    # Trouver le rang par numéro
    for rank in career_track.ranks:
        if rank.rank == 42:
            print(f"Rang: {rank.tier_type} {rank.rank_title.value} {rank.rank_tier.value}")
            print(f"Icône: {rank.rank_large_icon}")
            break
```

---

## 🟡 Résolution des chemins Inventory → URLs d'images

### Problème

L'API Economy retourne des **chemins JSON** comme :
- `Inventory/Spartan/BackdropImages/103-000-ui-background.json`
- `Inventory/Spartan/Emblems/104-001-olympus-stuck.json`

Ces chemins ne sont **pas des URLs d'images**. Il faut les résoudre.

### Solution 1 : Pattern Waypoint (rapide, sans appel API)

Pour certains items, le PNG existe à un chemin prédictible :

```python
# Emblems
# Input:  Inventory/Spartan/Emblems/{stem}.json + configuration_id
# Output: https://gamecms-hacs.svc.halowaypoint.com/hi/Waypoint/file/images/emblems/{stem}_{config_id}.png

# Backdrops (pattern simple - ne fonctionne PAS pour tous)
# Input:  Inventory/Spartan/BackdropImages/{stem}.json
# Output: https://gamecms-hacs.svc.halowaypoint.com/hi/Waypoint/file/images/backdrops/{stem}.png
```

⚠️ **Limitation** : Ce pattern ne fonctionne que pour ~60% des items.

### Solution 2 : API Progression (fiable, appel supplémentaire)

**Endpoint** :
```
GET https://gamecms-hacs.svc.halowaypoint.com/hi/progression/file/{INVENTORY_PATH}
```

Où `{INVENTORY_PATH}` = le chemin retourné par l'API Economy.

**Exemple** :
```
GET https://gamecms-hacs.svc.halowaypoint.com/hi/progression/file/Inventory/Spartan/BackdropImages/103-000-ui-background.json
```

**Réponse** :
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

**URL finale** :
```
https://gamecms-hacs.svc.halowaypoint.com/hi/images/file/progression/Backdrops/103-000-ui-background.png
```

### Code Python recommandé

```python
async def resolve_inventory_to_png(session, inventory_path: str, spartan_token: str, clearance_token: str) -> str | None:
    """Résout un chemin Inventory/*.json vers l'URL du PNG."""
    if not inventory_path:
        return None
    
    # Construire l'URL de l'API Progression
    path = inventory_path.lstrip("/")
    if not path.lower().startswith("inventory/"):
        return None
    
    url = f"https://gamecms-hacs.svc.halowaypoint.com/hi/progression/file/{path}"
    headers = {
        "Accept": "application/json",
        "X-343-Authorization-Spartan": spartan_token,
        "343-Clearance": clearance_token,
    }
    
    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
    
    # Extraire le chemin PNG
    png_path = (
        data.get("CommonData", {})
        .get("DisplayPath", {})
        .get("Media", {})
        .get("MediaUrl", {})
        .get("Path", "")
    )
    
    if not png_path:
        return None
    
    return f"https://gamecms-hacs.svc.halowaypoint.com/hi/images/file/{png_path.lstrip('/')}"
```

---

### 3. Backdrop / Item CMS (ancienne doc)

**Endpoint** :
```
GET https://gamecms-hacs.svc.halowaypoint.com/hi/progression/file/{ITEM_PATH}
```

Où `{ITEM_PATH}` = `Inventory/Spartan/BackdropImages/103-000-ui-background-e86f6dee.json`

**Code C# de référence** (UserContextManager.cs:548-575) :
```csharp
var backdrop = await SafeAPICall(async () => 
    await HaloClient.GameCmsGetItem(
        customizationResult.Result.Appearance.BackdropImagePath, 
        HaloClient.ClearanceToken
    ));

// L'URL PNG est dans :
string backdropPngUrl = backdrop.Result.ImagePath.Media.MediaUrl.Path;
```

**Structure de réponse attendue** :
```json
{
  "ImagePath": {
    "Media": {
      "MediaUrl": {
        "Path": "progression/Backdrops/103-000-ui-background-e86f6dee.png"
      }
    }
  }
}
```

**Workaround actuel** : Conversion heuristique du chemin JSON → PNG
```python
def _inventory_backdrop_to_waypoint_png(backdrop_path: str) -> str:
    # Input:  Inventory/Spartan/BackdropImages/103-000-ui-background-e86f6dee.json
    # Output: https://gamecms-hacs.svc.halowaypoint.com/hi/Waypoint/file/images/backdrops/103-000-ui-background-e86f6dee.png
    
    stem = backdrop_path.split("/Spartan/BackdropImages/", 1)[1].rsplit(".", 1)[0]
    return f"https://gamecms-hacs.svc.halowaypoint.com/hi/Waypoint/file/images/backdrops/{stem}.png"
```

---

### 4. Images CSR (Competitive Skill Rank)

**Endpoint public** (sans auth) :
```
GET https://gamecms-hacs.svc.halowaypoint.com/hi/Progression/file/Csr/Seasons/CsrSeason9-{tier}.png
```

**Tiers disponibles** :
- `bronze`, `silver`, `gold`, `platinum`, `diamond`, `onyx`
- Sous-tiers : `{tier}_{1-6}.png` (ex: `bronze_1.png`, `diamond_5.png`)
- Unranked : `unranked_{0-9}.png`

**Liste complète des rangs** (Configuration.cs:20-39) :
```python
CSR_RANKS = [
    "unranked_0", "unranked_1", "unranked_2", "unranked_3", "unranked_4",
    "unranked_5", "unranked_6", "unranked_7", "unranked_8", "unranked_9",
    "bronze_1", "bronze_2", "bronze_3", "bronze_4", "bronze_5", "bronze_6",
    "silver_1", "silver_2", "silver_3", "silver_4", "silver_5", "silver_6",
    "gold_1", "gold_2", "gold_3", "gold_4", "gold_5", "gold_6",
    "platinum_1", "platinum_2", "platinum_3", "platinum_4", "platinum_5", "platinum_6",
    "diamond_1", "diamond_2", "diamond_3", "diamond_4", "diamond_5", "diamond_6",
    "onyx_1",  # Onyx n'a qu'un seul niveau
]
```

---

### 5. Images de rang Career

**Endpoint** (via `get_image` de SPNKr) :
```
GET https://gamecms-hacs.svc.halowaypoint.com/hi/images/file/{RANK_ICON_PATH}
```

**Exemple** :
```python
# rank_large_icon = "career_rank/CelebrationMoment/42_Sergeant_Gold_II.png"
image_resp = await client.gamecms_hacs.get_image(rank.rank_large_icon)
image_bytes = image_resp.data
```

---

## 🏗️ Implémentation future suggérée

### Quand l'API Grunt sera publique

```python
# À ajouter dans spnkr/services/economy.py
async def get_player_career_rank(
    self,
    xuid: str | int,
    track_id: str = "careerRank1",
) -> JsonResponse[PlayerCareerRank]:
    """Get the career rank progression for a player.
    
    Args:
        xuid: The Xbox Live ID of the player.
        track_id: The reward track ID (default: "careerRank1").
    
    Returns:
        The player's career rank progression.
    """
    url = f"{_HOST}/hi/players/{wrap_xuid(xuid)}/rewardtracks/{track_id}"
    resp = await self._get(url)
    return JsonResponse(resp, lambda data: PlayerCareerRank(**data))
```

```python
# À ajouter dans spnkr/services/gamecms_hacs.py
async def get_item(
    self,
    item_path: str,
) -> JsonResponse[InGameItem]:
    """Get metadata for an in-game item.
    
    Args:
        item_path: The inventory path to the item JSON.
    
    Returns:
        The item metadata including image URLs.
    """
    # Normaliser le chemin
    path = item_path.lstrip("/")
    if not path.startswith("Inventory/"):
        path = f"Inventory/{path}"
    
    url = f"{_HOST}/hi/{path}"
    resp = await self._get(url)
    return JsonResponse(resp, lambda data: InGameItem(**data))
```

---

## 📚 Références

- **OpenSpartan Workshop** : https://github.com/OpenSpartan/openspartan-workshop
- **Fichier principal** : `src/OpenSpartan.Workshop/Core/UserContextManager.cs`
- **SPNKr (Python)** : https://github.com/acurtis166/SPNKr
- **Grunt (C# - privé)** : Package NuGet `Den.Dev.Grunt`

---

## 📝 Notes

1. **Hero Rank** (272) : Le rang maximum, traité différemment (pas de `+1` pour l'affichage)
2. **Bug connu** (ligne 327-332 C#) : `career_rank/CelebrationMoment/219_Cadet_Onyx_III.png` doit être corrigé en `19_Cadet_Onyx_III.png`
3. **Tokens requis** : Spartan Token + Clearance Token pour les appels Economy
4. **Cache recommandé** : Les métadonnées de rang changent rarement, cacher pour 24h+
