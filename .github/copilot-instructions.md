# Instructions pour GitHub Copilot & Assistants IA

Ce fichier définit les conventions et règles à suivre lors de modifications sur ce projet.

---

## 🤖 Workflow d'interaction IA

### Avant toute modification

1. **Analyser la demande** : Reformuler pour confirmer la compréhension
2. **Explorer le contexte** : Lire les fichiers concernés, comprendre l'existant
3. **Proposer un plan** : Lister les étapes avant d'implémenter
4. **Valider avec l'utilisateur** : Attendre le "go" avant les modifications majeures
5. **Implémenter par phases** : Découper en commits logiques

### Structure d'une réponse idéale

```markdown
## 🎯 Compréhension de la demande
[Reformulation en 1-2 phrases]

## 🔍 Analyse de l'existant
- Fichiers impactés : ...
- Dépendances : ...
- Risques identifiés : ...

## 📋 Plan d'implémentation
1. [ ] Étape 1 - Description
2. [ ] Étape 2 - Description
3. [ ] Étape 3 - Description

## ⚠️ Points de vigilance
- ...

Tu veux que je procède ?
```

### Bonnes pratiques

| ✅ Faire | ❌ Éviter |
|----------|-----------|
| Demander des précisions si ambigu | Deviner les intentions |
| Proposer plusieurs options | Imposer une solution unique |
| Expliquer les choix techniques | Modifier silencieusement |
| Tester avant de valider | Supposer que ça fonctionne |
| Commiter par petits incréments | Un gros commit monolithique |

### Questions à poser si contexte insuffisant

- "Quel est le comportement attendu ?"
- "Y a-t-il des contraintes de performance ?"
- "Faut-il maintenir la rétrocompatibilité ?"
- "Préfères-tu une solution simple ou extensible ?"
- "Dois-je ajouter des tests pour cette feature ?"

---

## 🎯 Contexte du projet

**OpenSpartan Graph** est un dashboard Streamlit pour analyser les statistiques Halo Infinite.

- **Stack** : Python 3.10+, Streamlit, SQLite, SPNKr (API Halo)
- **Langue UI** : Français (traductions dans `src/ui/translations.py`)
- **Base de données** : SQLite avec tables `MatchStats`, `XuidAliases`, `SyncMeta`, `HighlightEvents`

---

## 📁 Architecture

```
src/
├── config.py          # Configuration centralisée (constantes, chemins)
├── models.py          # Dataclasses uniquement (pas de logique)
├── db/                # Accès base de données
│   ├── loaders.py     # Chargement données + cache Streamlit
│   ├── parsers.py     # Parsing JSON des matchs
│   └── queries.py     # Requêtes SQL brutes
├── analysis/          # Fonctions d'analyse (pandas)
│   ├── filters.py     # Filtres playlists/modes
│   ├── stats.py       # Calculs statistiques
│   └── sessions.py    # Détection sessions de jeu
├── ui/                # Helpers interface
│   ├── translations.py # Traductions FR (PLAYLIST_FR, PAIR_FR)
│   ├── aliases.py     # Gestion alias joueurs
│   └── settings.py    # Paramètres utilisateur
└── visualization/     # Graphiques (Altair/Plotly)
```

---

## ✅ Conventions de code

### Python

- **Type hints** obligatoires sur toutes les fonctions publiques
- **Docstrings** en français pour les fonctions principales
- **Imports** : `from __future__ import annotations` en premier
- **Formatage** : Black + isort + ruff
- **Dataclasses** pour les modèles de données (pas de dicts anonymes)

```python
# ✅ Bon
def compute_kd_ratio(kills: int, deaths: int) -> float:
    """Calcule le ratio kills/deaths."""
    if deaths == 0:
        return float(kills)
    return kills / deaths

# ❌ Mauvais
def compute_kd_ratio(kills, deaths):
    return kills / deaths if deaths else kills
```

### Streamlit

- **Cache** : Utiliser `@st.cache_data` pour les fonctions de chargement
- **Session state** : Préfixer les clés avec le contexte (`filter_`, `ui_`, `sync_`)
- **Sidebar** : Filtres et paramètres dans la sidebar, contenu principal au centre
- **Rerun** : Éviter les `st.rerun()` sauf nécessité absolue

### SQL / Base de données

- **Paramètres** : Toujours utiliser des placeholders `?` (jamais de f-strings)
- **Transactions** : Commit explicite après les modifications
- **Nouvelles tables** : Documenter dans le README section "Tables de base de données"

```python
# ✅ Bon
cur.execute("SELECT * FROM MatchStats WHERE match_id = ?", (match_id,))

# ❌ Mauvais (injection SQL)
cur.execute(f"SELECT * FROM MatchStats WHERE match_id = '{match_id}'")
```

---

## 🌍 Traductions

### Ajouter une nouvelle playlist

1. Ajouter dans `PLAYLIST_FR` de `src/ui/translations.py`
2. Mettre à jour `Playlist_modes_translations.json`

```python
PLAYLIST_FR: dict[str, str] = {
    "New Playlist": "Nouvelle playlist",
    # ...
}
```

### Ajouter un nouveau mode de jeu

1. Ajouter dans `PAIR_FR` avec le format `"Prefix:Mode on Map": "Traduction"`
2. Ajouter aussi le fallback générique `"Prefix:Mode": "Traduction"`

```python
PAIR_FR: dict[str, str] = {
    # Fallback générique
    "Arena:NewMode": "Arène : Nouveau mode",
    # Entrées spécifiques
    "Arena:NewMode on Aquarius": "Arène : Nouveau mode",
    "Arena:NewMode on Bazaar": "Arène : Nouveau mode",
}
```

---

## 🔄 Sync & Delta

### Mode Delta

Le mode `--delta` ne récupère que les nouveaux matchs depuis la dernière sync.

- **Table `SyncMeta`** : Stocke `last_sync`, `last_match_id`, `total_matches`
- **Table `XuidAliases`** : Mapping XUID → Gamertag (auto-peuplé)

### Ajouter une métadonnée de sync

```python
def update_sync_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    cur = con.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        INSERT INTO SyncMeta (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
    """, (key, value, now, value, now))
    con.commit()
```

---

## 🧪 Tests

### Conventions

- Fichiers dans `tests/test_*.py`
- Classes de test préfixées `Test*`
- Un fichier de test par module ou feature
- Mocks pour les appels API/DB externes

### Lancer les tests

```bash
pytest                          # Tous les tests
pytest tests/test_delta_sync.py # Tests spécifiques
pytest --cov=src               # Avec couverture
```

### Structure d'un test

```python
class TestMyFeature:
    """Tests pour ma fonctionnalité."""

    def test_normal_case(self):
        """Test avec des valeurs normales."""
        result = my_function(10, 5)
        assert result == expected

    def test_edge_case(self):
        """Test avec cas limites."""
        assert my_function(0, 0) is None
```

---

## 📝 Commits

### Format Conventional Commits

```
<type>(<scope>): <description>

[body optionnel]
```

### Types autorisés

| Type | Description |
|------|-------------|
| `feat` | Nouvelle fonctionnalité |
| `fix` | Correction de bug |
| `docs` | Documentation |
| `refactor` | Refactoring sans changement fonctionnel |
| `test` | Ajout/modification de tests |
| `chore` | Maintenance (deps, config) |

### Exemples

```
feat(ui): ajouter indicateur de sync dans la sidebar
fix(filters): inclure Big Team Battle dans les playlists autorisées
docs: mettre à jour README avec instructions delta sync
test(translations): ajouter tests pour translate_pair_name
```

---

## 🚫 À éviter

1. **Ne pas** modifier `streamlit_app.py` sans vérifier l'impact sur le rerun
2. **Ne pas** ajouter de `print()` — utiliser `st.info()` ou logging
3. **Ne pas** hardcoder des chemins Windows — utiliser `Path` de pathlib
4. **Ne pas** créer de nouvelles dépendances sans les ajouter à `pyproject.toml`
5. **Ne pas** modifier les tables DB existantes sans migration

---

## 📋 Checklist avant PR

- [ ] Tests passent (`pytest`)
- [ ] Pas d'erreurs de type (`pyright` ou Pylance)
- [ ] Traductions FR à jour si nouvelle UI
- [ ] README mis à jour si nouvelle feature
- [ ] Commit message au format Conventional Commits

---

## 🔧 Configuration IDE recommandée

### VS Code settings.json

```json
{
  "python.analysis.typeCheckingMode": "basic",
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

---

## 💡 Ressources

- [SPNKr Documentation](https://github.com/acurtis166/SPNKr)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Halo Infinite API (non officielle)](https://den.dev/blog/halo-infinite-api-authentication/)

---

## 🧠 Conseils de prompt engineering (pour l'utilisateur)

### Structurer ses demandes

```markdown
# ✅ Bon prompt
"Ajouter un filtre par carte dans la sidebar.
- Dropdown multi-select avec toutes les cartes du DataFrame
- Persister la sélection dans session_state
- Appliquer le filtre avant les calculs de stats"

# ❌ Prompt vague
"Ajouter un filtre par carte"
```

### Fournir du contexte

- **Fichiers concernés** : "Dans `streamlit_app.py`, fonction `_render_filters()`..."
- **Comportement actuel** : "Actuellement, seul le filtre playlist existe..."
- **Résultat attendu** : "Je veux pouvoir filtrer par Aquarius, Bazaar, etc."

### Mots-clés efficaces

| Mot-clé | Effet |
|---------|-------|
| "Analyse d'abord..." | Force l'exploration avant action |
| "Propose un plan..." | Évite l'implémentation directe |
| "Étape par étape..." | Découpe en phases validables |
| "Comme dans [fichier]..." | Référence un pattern existant |
| "Sans casser..." | Impose la rétrocompatibilité |
| "Avec tests..." | Inclut les tests unitaires |

### Anti-patterns à éviter

1. ❌ Demandes trop larges : "Refais tout le dashboard"
2. ❌ Manque de critères : "Améliore les perfs" (quelles métriques ?)
3. ❌ Contradictions implicites : "Simple mais extensible et performant"
4. ❌ Validation post-hoc : Valider avant, pas après les modifs massives
