# 🎮 OpenSpartan Graph

> **Dashboard interactif et CLI pour analyser vos statistiques Halo Infinite**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📑 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Nouveautés v2.0](#-nouveautés-v20---delta-sync-pipeline)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
  - [Dashboard](#dashboard-recommandé)
  - [Sync incrémental (Delta)](#-sync-incrémental-delta)
  - [Rafraîchir la DB](#rafraîchir-la-db-au-lancement-spnkr)
  - [Réparer les gamertags](#réparer-les-gamertags-aliases-via-film-roster)
- [Configuration](#️-configuration)
- [Architecture](#-architecture)
- [Tests](#-tests)
- [Docker](#-docker)
- [Contribution](#-contribution)

---

## ✨ Fonctionnalités

### Core
- 📊 **Dashboard interactif** — Visualisez vos stats en temps réel avec Streamlit
- 📈 **Graphiques détaillés** — Évolution frags/morts/assistances, précision, durée de vie moyenne, séries de frags
- 🗺️ **Analyse par carte** — Performance détaillée sur chaque map
- 👥 **Analyse des coéquipiers** — Statistiques avec vos amis (même équipe ou adversaires)
- 🎯 **Sessions de jeu** — Détection automatique des sessions avec métriques

### Export & Personnalisation
- 🖼️ **Export PNG** — Générez des graphiques statiques via CLI
- 🎨 **Thème Halo** — Interface inspirée de Halo Waypoint
- 🌍 **Traductions FR** — Interface et modes de jeu traduits en français (313+ modes)

---

## 🆕 Nouveautés v2.0 - Delta Sync Pipeline

### ⚡ Sync incrémental (Delta Mode)

Plus besoin de tout resynchroniser ! Le mode delta ne récupère que les **nouveaux matchs** :

```bash
# Sync rapide (delta) - seulement les nouveaux matchs
python openspartan_launcher.py refresh --player MonGamertag --delta

# Sync complet (si besoin)
python openspartan_launcher.py refresh --player MonGamertag
```

### 📋 Tables de métadonnées

| Table | Description |
|-------|-------------|
| `SyncMeta` | Suivi des synchronisations (dernière sync, compteurs) |
| `XuidAliases` | Mapping XUID → Gamertag (auto-peuplé depuis les matchs) |
| `HighlightEvents` | Événements marquants (frags, morts, médailles) |

### 🎯 Highlight Events par défaut

Les highlight events sont maintenant extraits automatiquement lors de l'import, permettant d'afficher :
- Les kills/deaths remarquables
- Les médailles obtenues
- Les séquences de frags

### 🔄 Indicateur de sync dans la sidebar

La sidebar affiche maintenant :
- ⏱️ Date de dernière synchronisation
- 📊 Nombre de matchs synchronisés
- 🔘 Boutons **Sync** (delta) et **Full** (complet)

---

## 📋 Prérequis

- **Windows 10/11** (ou Linux/macOS via Docker)
- **Python 3.10+** (recommandé: 3.11 ou 3.12)
- **Compte Azure AD**
- **SPNKr** API Halo Infinite

---

## 📦 Installation

### Installation rapide

```bash
# Cloner le projet
git clone https://github.com/username/openspartan-graph.git
cd openspartan-graph

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows)
.venv\Scripts\activate

# Activer l'environnement (Linux/macOS)
source .venv/bin/activate

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

### Configuration SPNKr (API Halo)

1. Copier `.env.example` → `.env.local`
2. Configurer vos tokens Azure :

```env
SPNKR_AZURE_CLIENT_ID=votre_client_id
SPNKR_AZURE_CLIENT_SECRET=votre_secret
SPNKR_AZURE_REDIRECT_URI=https://localhost
SPNKR_OAUTH_REFRESH_TOKEN=votre_refresh_token
```

3. Récupérer le refresh token :

```bash
python scripts/spnkr_get_refresh_token.py
```

---

## 🎮 Utilisation

### Dashboard (recommandé)

Le mode de lancement recommandé est le **lanceur Python unique** :

```bash
# Mode interactif (questions automatiques)
python openspartan_launcher.py

# Lancer directement le dashboard
python openspartan_launcher.py run

# Afficher l'aide complète
python openspartan_launcher.py --help
```

### ⚡ Sync incrémental (Delta)

```bash
# Sync rapide (delta) - récupère uniquement les nouveaux matchs
python openspartan_launcher.py refresh --player MonGamertag --delta

# Sync complet avec highlight events
python openspartan_launcher.py refresh --player MonGamertag --patch-highlight-events

# Sync + lancer le dashboard
python openspartan_launcher.py run+refresh --player MonGamertag --delta
```

#### Options de synchronisation

| Option | Description | Défaut |
|--------|-------------|--------|
| `--delta` | Mode incrémental (nouveaux matchs seulement) | Non |
| `--max-matches N` | Limite de matchs à récupérer | 50 |
| `--match-type` | Type de matchs (`all`, `matchmaking`, `custom`) | matchmaking |
| `--patch-highlight-events` | Extraire les highlight events | Non |
| `--no-assets` | Ne pas télécharger les assets (plus rapide) | Non |

### Rafraîchir la DB au lancement (SPNKr)

```bash
# Premier lancement (bootstrap complet)
python openspartan_launcher.py run+refresh --player MonGamertag

# Lancements suivants (delta)
python openspartan_launcher.py run+refresh --player MonGamertag --delta
```

### Réparer les gamertags (aliases) via film roster

Quand les gamertags dans `HighlightEvents` sont corrompus :

```bash
# Répare le match le plus récent
python openspartan_launcher.py repair-aliases --db data/spnkr_gt_MonGamertag.db --latest

# Répare tous les matchs
python openspartan_launcher.py repair-aliases --db data/spnkr_gt_MonGamertag.db --all-matches
```

### CLI (génération PNG)

```bash
python openspartan_graph.py --db "data/spnkr_gt_MonGamertag.db" --last 80 --out "out/stats.png"
```

---

## ⚙️ Configuration

### Filtres (sidebar)

| Option | Description | Défaut |
|--------|-------------|--------|
| **Inclure Firefight** | Afficher les parties Firefight (PvE) | ❌ |
| **Restreindre playlists** | Limiter à Quick Play, Ranked, BTB | ❌ |

### Playlists supportées

Toutes les playlists sont maintenant affichées par défaut, incluant :
- Quick Play, Ranked Arena, Ranked Slayer
- **Big Team Battle** (toutes variantes)
- Firefight, Super Fiesta, Team Snipers
- Modes communautaires, événements spéciaux

### Variables d'environnement

| Variable | Description |
|----------|-------------|
| `OPENSPARTAN_DB_PATH` | Chemin vers la base de données |
| `OPENSPARTAN_DB_READONLY` | Mode lecture seule (Docker) |
| `SPNKR_PLAYER` | Joueur par défaut pour le refresh |

---

## 🏗️ Architecture

```
openspartan-graph/
├── src/                        # Code source modulaire
│   ├── config.py              # Configuration centralisée
│   ├── models.py              # Modèles de données (dataclasses)
│   ├── db/                    # Accès base de données
│   │   ├── connection.py      # Gestion connexions SQLite
│   │   ├── loaders.py         # Chargement des données + SyncMeta
│   │   ├── parsers.py         # Parsing JSON des matchs
│   │   ├── profiles.py        # Gestion profils joueurs
│   │   └── queries.py         # Requêtes SQL
│   ├── analysis/              # Fonctions d'analyse
│   │   ├── filters.py         # Filtres playlists (Big Team Battle inclus)
│   │   ├── killer_victim.py   # Analyse confrontations
│   │   ├── maps.py            # Stats par carte
│   │   ├── sessions.py        # Détection sessions
│   │   └── stats.py           # Calculs statistiques
│   ├── ui/                    # Helpers interface utilisateur
│   │   ├── aliases.py         # Gestion des alias joueurs
│   │   ├── translations.py    # Traductions FR (313+ modes)
│   │   ├── medals.py          # Affichage médailles
│   │   ├── settings.py        # Paramètres utilisateur (dataclass)
│   │   ├── components/        # Composants réutilisables
│   │   │   └── performance.py # Score de performance sessions
│   │   └── pages/             # Pages du dashboard (modulaires)
│   │       ├── session_compare.py  # Comparaison de sessions
│   │       ├── timeseries.py       # Séries temporelles
│   │       ├── win_loss.py         # Victoires/Défaites
│   │       ├── match_history.py    # Historique des parties
│   │       ├── teammates.py        # Analyse coéquipiers
│   │       ├── citations.py        # Citations & Médailles
│   │       └── settings.py         # Page Paramètres
│   └── visualization/         # Génération des graphiques
│       ├── distributions.py   # Histogrammes, box plots
│       ├── maps.py            # Heatmaps cartes
│       ├── theme.py           # Thème Halo
│       └── timeseries.py      # Graphiques temporels
├── scripts/                    # Scripts utilitaires
│   ├── spnkr_import_db.py     # Import SPNKr avec delta
│   ├── spnkr_get_refresh_token.py  # Auth Azure
│   └── prefetch_profile_assets.py  # Préchargement assets
├── static/                     # Fichiers statiques
│   ├── styles.css             # Thème CSS Halo Waypoint
│   └── medals/                # Icônes médailles
├── tests/                      # Suite de tests pytest
│   ├── test_delta_sync.py     # Tests sync delta
│   ├── test_analysis.py       # Tests analyse
│   └── test_models.py         # Tests modèles
├── data/                       # Données locales (gitignored)
│   ├── cache/                 # Cache API et assets
│   └── spnkr_gt_*.db          # Bases de données joueurs
├── streamlit_app.py           # Point d'entrée dashboard
├── openspartan_launcher.py    # Lanceur CLI unifié
└── pyproject.toml             # Configuration projet
```

### Tables de base de données

| Table | Description |
|-------|-------------|
| `MatchStats` | Statistiques des matchs (JSON compressé) |
| `HighlightEvents` | Événements marquants extraits |
| `XuidAliases` | Mapping XUID → Gamertag |
| `SyncMeta` | Métadonnées de synchronisation |
| `Playlists` | Informations playlists |
| `PlaylistMapModePairs` | Modes de jeu |
| `Maps`, `GameVariants` | Assets de jeu |

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest

# Avec couverture
pytest --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_delta_sync.py -v
pytest tests/test_analysis.py -v

# Tests rapides (sans couverture)
pytest -x --tb=short
```

### Couverture actuelle

| Module | Couverture |
|--------|------------|
| `src/ui/translations.py` | 100% |
| `src/analysis/filters.py` | 95% |
| `src/db/loaders.py` | 85% |

---

## 🐳 Docker

### Docker Compose (recommandé)

```bash
# Démarrer
docker compose up --build

# Accéder au dashboard
open http://localhost:8501
```

### Configuration Docker

```yaml
# docker-compose.yml
services:
  openspartan:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/data:ro
      - ./appdata:/appdata
    environment:
      - OPENSPARTAN_DB=/data/spnkr_gt_MonGamertag.db
      - OPENSPARTAN_DB_READONLY=1
```

---

## 📝 Changelog

### v2.0.0 (2026-01-22)

#### ✨ Nouvelles fonctionnalités
- **Delta Sync** : Mode `--delta` pour synchronisation incrémentale
- **Tables SyncMeta/XuidAliases** : Suivi des syncs et mapping gamertags
- **Highlight Events par défaut** : Extraction automatique à l'import
- **Indicateur sync sidebar** : Affichage dernière sync + boutons Sync/Full
- **Traductions complètes** : 313 modes de jeu traduits en français

#### 🔧 Améliorations UX
- Filtres déplacés dans la sidebar (plus accessible)
- Big Team Battle ajouté aux playlists autorisées
- `restrict_playlists=False` par défaut (tous les matchs affichés)

#### 🐛 Corrections
- Fix affichage 281/955 matchs (filtres trop restrictifs)
- Fix gamertags corrompus via repair-aliases

---

## 🤝 Contribution

Les contributions sont les bienvenues !

1. Fork le projet
2. Créer une branche (`git checkout -b feature/ma-feature`)
3. Commit (`git commit -m 'feat: ajout ma feature'`)
4. Push (`git push origin feature/ma-feature`)
5. Ouvrir une Pull Request

### Conventions

- **Commits** : Format [Conventional Commits](https://www.conventionalcommits.org/)
- **Code** : Black + isort + ruff
- **Tests** : pytest avec couverture > 80%

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

**Fait avec ❤️ pour la communauté Halo**
