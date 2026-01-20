# 🎮 OpenSpartan Graph

> **Dashboard interactif et CLI pour analyser vos statistiques Halo Infinite**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Fonctionnalités

- 📊 **Dashboard interactif** — Visualisez vos stats en temps réel avec Streamlit
- 📈 **Graphiques détaillés** — Évolution frags/morts/assistances, précision, durée de vie moyenne, séries de frags
- 🗺️ **Analyse par carte** — Performance détaillée sur chaque map
- 👥 **Analyse des coéquipiers** — Statistiques avec vos amis (même équipe ou adversaires)
- 🎯 **Sessions de jeu** — Détection automatique des sessions avec métriques
- 🖼️ **Export PNG** — Générez des graphiques statiques via CLI
- 🎨 **Thème Halo** — Interface inspirée de Halo Waypoint

## 📋 Prérequis

- **Windows 10/11**
- **Python 3.10+** (recommandé: 3.11 ou 3.12)
- **[OpenSpartan Workshop](https://github.com/OpenSpartan/openspartan-workshop)** installé et synchronisé

## 📦 Assets offline (icônes)

### Icônes de médailles (Halo Infinite)

Par défaut, l'app lit les icônes de médailles depuis le cache OpenSpartan.Workshop (dans ton profil Windows).
Pour rendre le projet autonome/offline, copie les PNG du cache vers le repo :

```bash
python scripts/sync_medal_icons.py
```

- Destination : `static/medals/icons/<NameId>.png`
- Ensuite, l'UI utilisera automatiquement ces icônes locales (fallback vers le cache OpenSpartan si besoin).

## 🚀 Installation

### Installation rapide

```bash
# Cloner le projet
git clone https://github.com/username/openspartan-graph.git
cd openspartan-graph

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement
.venv\Scripts\activate

# Installer les dépendances
pip install -e .
```

### Installation développeur

```bash
# Avec les outils de dev (tests, linting, typing)
pip install -e ".[dev]"

# Avec le CLI matplotlib
pip install -e ".[cli]"

# Tout installer
pip install -e ".[all]"
```

## 🎮 Utilisation

### Dashboard (recommandé)

**Le plus simple :** double-cliquez sur `run_dashboard.bat`

Ou en ligne de commande :

```bash
# Via le launcher
python run_dashboard.py

# Ou directement Streamlit
streamlit run streamlit_app.py
```

Le dashboard s'ouvre automatiquement dans votre navigateur.

### Rafraîchir la DB au lancement (SPNKr)

Si vous utilisez l'import SPNKr ([scripts/spnkr_import_db.py](scripts/spnkr_import_db.py)), vous pouvez rafraîchir la base avant d'ouvrir Streamlit.

- Pré-requis: avoir l'auth SPNKr configurée (ex: `SPNKR_OAUTH_REFRESH_TOKEN` dans `.env.local`) et un joueur cible.
- Définissez le joueur via `SPNKR_PLAYER` (env) ou `--refresh-player`.

Exemple (recommandé, mode minimal fiable):

```bash
python run_dashboard.py --refresh-spnkr --refresh-no-assets
```

Au premier lancement (si `data/spnkr.db` n'existe pas ou est vide), le launcher fait automatiquement une **construction complète** (bootstrap) avec un `--max-matches` élevé et `--match-type all`.
Ensuite, les lancements suivants font un refresh plus léger.

Options utiles:

- `--refresh-max-matches 50` (défaut: 50)
- `--refresh-bootstrap-max-matches 2000` (défaut: 2000)
- `--refresh-match-type matchmaking` (défaut: matchmaking)
- `--refresh-bootstrap-match-type all` (défaut: all)
- `--refresh-out-db data/spnkr.db` (défaut: data/spnkr.db)

### Changer le joueur par défaut (Gamertag / XUID)

Le projet est configuré avec des valeurs par défaut pour simplifier l'usage en local.

- **Dans le code (valeurs en dur)**: modifie `DEFAULT_PLAYER_GAMERTAG` et `DEFAULT_PLAYER_XUID` dans [src/config.py](src/config.py).
- **Dans le launcher Windows (valeurs en dur)**: modifie `DEFAULT_GAMERTAG` et `DEFAULT_XUID` dans [run_dashboard.bat](run_dashboard.bat).
- **Au lancement (sans toucher au code)**:
  - `SPNKR_PLAYER` (env) permet d'override le joueur ciblé par le refresh SPNKr.
  - Le chemin DB utilisé par le dashboard peut être forcé via `OPENSPARTAN_DB_PATH` (ou `OPENSPARTAN_DB`).

Sous Windows, le launcher [run_dashboard.bat](run_dashboard.bat) a un défaut (actuellement `JGtm`) et tu peux l'override via `SPNKR_PLAYER`.

### CLI (génération PNG)

```bash
python openspartan_graph.py --db "%LOCALAPPDATA%\OpenSpartan.Workshop\data\<votre_xuid>.db" --last 80 --out "out\stats.png"
```

Options disponibles :

| Option | Description |
|--------|-------------|
| `--db` | Chemin vers la base de données SQLite |
| `--last N` | Limiter aux N derniers matchs |
| `--out` | Chemin du fichier PNG de sortie |

## 🗄️ Base de données

Par défaut, l'application détecte automatiquement la DB la plus récente dans :

```
%LOCALAPPDATA%\OpenSpartan.Workshop\data\*.db
```

Vous pouvez aussi spécifier un chemin personnalisé dans la sidebar du dashboard.

### Import alternatif (SPNKr)

Si OpenSpartan Workshop est instable, vous pouvez générer une DB compatible via SPNKr (wrapper API Halo Infinite) :

```bash
pip install "spnkr @ git+https://github.com/acurtis166/SPNKr.git"

# Tokens (option simple)
# 1) Copie `.env.example` -> `.env` (ou `.env.local.example` -> `.env.local`)
# 2) Remplis SPNKR_SPARTAN_TOKEN et SPNKR_CLEARANCE_TOKEN
#    (le script charge automatiquement `.env.local` puis `.env` si présents)

python scripts/spnkr_import_db.py --out-db data\spnkr.db --player <xuid_ou_gamertag> --max-matches 200 --resume

Astuce (import minimal, plus robuste) :

```bash
python scripts/spnkr_import_db.py --out-db data\spnkr.db --player <xuid_ou_gamertag> --max-matches 50 --resume --no-assets
```
```

#### Option Azure (recommandée)

La doc officielle SPNKr propose un flow Azure AD qui évite de récupérer `343-clearance` à la main.

1) Dans Azure AD, crée une App Registration, ajoute `https://localhost` en Redirect URI (type Web), puis génère un client secret.

Guide anti-galère (portail Azure) :
- Va sur `portal.azure.com`
- Dans la barre de recherche du haut, tape **App registrations** (ou **Inscriptions d’applications**)
- Clique **New registration**
- **Supported account types** : choisis l’option qui inclut **personal Microsoft accounts**
- **Redirect URI** : Type **Web**, URL `https://localhost`
- Ensuite: **Gérer** → **Certificates & secrets** → **New client secret** → copie la **Value** (pas l’ID)

Sécurité :
- Ne commit jamais `SPNKR_AZURE_CLIENT_SECRET` ni `SPNKR_OAUTH_REFRESH_TOKEN`.
- Utilise `.env.local` (ignoré par git) pour stocker ces valeurs.

2) Mets ces valeurs dans `.env.local` :

```text
SPNKR_AZURE_CLIENT_ID=...
SPNKR_AZURE_CLIENT_SECRET=...
SPNKR_AZURE_REDIRECT_URI=https://localhost
```

3) Récupère une fois ton refresh token :

```bash
python scripts/spnkr_get_refresh_token.py
```

Le script affiche une URL `login.live.com`. Ouvre-la, connecte-toi, puis à la fin copie l'URL `https://localhost/?code=...` depuis la barre d'adresse.
Note: la page `https://localhost` affiche souvent une erreur (pas de serveur local). C'est normal : ce qui compte c'est l'URL et le paramètre `code=`.

Ensuite relance :

```bash
python scripts/spnkr_get_refresh_token.py --auth-code "https://localhost/?code=..."
```

Le script écrit automatiquement `SPNKR_OAUTH_REFRESH_TOKEN` dans `.env.local` (tu peux désactiver avec `--no-write-env-local`).

Ensuite, relance l’import normalement (le script utilisera Azure automatiquement si ces variables sont présentes).

FAQ (Azure)
- `error=unauthorized_client` / "client does not have a secret configured" : tu n'as pas créé de **Client secret** (ou tu as copié le mauvais champ). Va dans **Certificates & secrets** → **New client secret** puis copie la **Value** (pas le Secret ID) dans `SPNKR_AZURE_CLIENT_SECRET`.
- `unauthorized_client` / "not enabled for consumers" : ton App Registration n'autorise pas les comptes Microsoft personnels. Dans **App registrations** → (ton app) → **Supported account types**, choisis une option incluant **personal Microsoft accounts** (ou modifie le manifest `signInAudience` vers `AzureADandPersonalMicrosoftAccount`).
- `invalid_client` / "client_secret is not valid" : le secret ne correspond pas au client id (souvent 2 apps différentes) ou le secret a expiré. Regénère un secret (copie la **Value**) et regénère un nouveau `code=` (un code OAuth est à usage unique et peut expirer vite). Le helper tente un fallback via endpoint OAuth v2 (consumers) si `login.live.com` refuse le secret.
- Si tu ne vois jamais `code=` dans l'URL de `https://localhost` : vérifie que le redirect URI configuré dans Azure est exactement `https://localhost` (type Web), et qu'il correspond à `SPNKR_AZURE_REDIRECT_URI`.

Ensuite, pointez la sidebar du dashboard sur `data\spnkr.db`.

## ⚡ Performance (démarrage / rerun)

Streamlit relance le script à chaque interaction (rerun). Pour diagnostiquer un démarrage un peu long :

- Active **Mode perf** dans la sidebar pour afficher les timings par section (CSS, sidebar, chargement DB, etc.).
- Utilise **Vider caches** si la DB a changé en dehors de l'app et que tu veux forcer un rechargement.
- Le scan des DB locales est volontairement mis sous cache (TTL court) pour éviter des accès disque trop fréquents.

## 🧪 Tests

```bash
# Lancer tous les tests
pytest

# Avec couverture
pytest --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_models.py -v
```

## 🐳 Docker

Le container ne peut pas « découvrir » automatiquement la DB Windows (pas de `LOCALAPPDATA`).
Monte donc ton fichier `.db` en volume et fournis son chemin via `OPENSPARTAN_DB`.

Astuce : tu peux monter la DB en lecture seule (`:ro`). L'app détecte ça et bascule en SQLite read-only
automatiquement, ou via `OPENSPARTAN_DB_READONLY=1`.

### Docker Compose (recommandé)

1) Place ta DB dans `./data/openspartan.db` (ou adapte le chemin)

2) (Optionnel) Pour persister profils/alias Streamlit entre redémarrages, crée un dossier `./appdata`.

3) Lance :

```bash
docker compose up --build
```

Puis ouvre `http://localhost:8501`.

### Docker (sans compose)

```bash
docker build -t openspartan-graph .

docker run --rm -p 8501:8501 \
	-e OPENSPARTAN_DB=/data/openspartan.db \
	-e OPENSPARTAN_DB_READONLY=1 \
	-e OPENSPARTAN_PROFILES_PATH=/appdata/db_profiles.json \
	-e OPENSPARTAN_ALIASES_PATH=/appdata/xuid_aliases.json \
	-v "%CD%\data:/data:ro" \
	-v "%CD%\appdata:/appdata" \
	openspartan-graph
```

## 📁 Structure du projet

```
openspartan-graph/
├── src/                    # Code source modulaire
│   ├── config.py          # Configuration centralisée
│   ├── models.py          # Modèles de données (dataclasses)
│   ├── db/                # Accès base de données
│   ├── analysis/          # Fonctions d'analyse
│   ├── visualization/     # Génération des graphiques
│   └── ui/                # Helpers interface utilisateur
├── static/
│   └── styles.css         # Thème CSS Halo Waypoint
├── tests/                  # Suite de tests pytest
├── streamlit_app.py       # Point d'entrée dashboard
├── openspartan_graph.py   # Point d'entrée CLI
├── run_dashboard.py       # Launcher avec port auto
├── run_dashboard.bat      # Script Windows
└── pyproject.toml         # Configuration projet
```

## ⚙️ Configuration

### Filtres par défaut

- **Playlists** : Quick Play, Ranked Slayer, Ranked Arena
- **Firefight** : Exclu par défaut (configurable)
- **Sessions** : Détection avec seuil de 30 minutes d'inactivité

Ces options sont modifiables dans la sidebar du dashboard.

## 📝 Notes

- Certaines stats (temps joué, précision) peuvent être absentes sur d'anciens matchs
- Les métriques "par minute" ignorent automatiquement les matchs sans durée valide
- Le système d'alias permet de renommer les joueurs (stocké dans `aliases.json`)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une PR.

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

**Fait avec ❤️ pour la communauté Halo**
