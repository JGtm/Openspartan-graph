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

### Docker Compose (recommandé)

1) Place ta DB dans `./data/openspartan.db` (ou adapte le chemin)

2) Lance :

```bash
docker compose up --build
```

Puis ouvre `http://localhost:8501`.

### Docker (sans compose)

```bash
docker build -t openspartan-graph .

docker run --rm -p 8501:8501 \
	-e OPENSPARTAN_DB=/data/openspartan.db \
	-v "%CD%\data:/data:ro" \
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
