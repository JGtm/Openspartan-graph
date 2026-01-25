# Score de performance RELATIF (v3)

Cette page documente le **score de performance relatif** affiché dans l'application.

## Fichiers sources

- Configuration centralisée : [src/analysis/performance_config.py](../src/analysis/performance_config.py)
- Algorithme de calcul : [src/analysis/performance_score.py](../src/analysis/performance_score.py)
- Script de migration historique : [scripts/compute_historical_performance.py](../scripts/compute_historical_performance.py)
- UI (comparaison sessions) : [src/ui/pages/session_compare.py](../src/ui/pages/session_compare.py)

---

## Objectif

Fournir une **note personnalisée (0–100)** qui compare ta performance **à ton propre historique**.

> **Pourquoi relatif ?** Un joueur occasionnel avec un K/D de 0.8 qui fait une partie à 1.2 a potentiellement fourni un meilleur effort qu'un joueur vétéran qui fait 1.5 (sa moyenne habituelle). Le score relatif récompense la **progression personnelle**.

---

## Philosophie

### Problèmes des scores absolus

1. **Injustice entre niveaux** : Un débutant ne peut jamais atteindre 100
2. **Plafond de verre** : Les bons joueurs stagnent autour de 80-90
3. **Pas de contexte** : Un match "moyen" n'a pas le même sens pour tout le monde

### Solution : le percentile relatif

Pour chaque match, on compare tes stats **aux matchs précédents** :

- **KPM** (Kills Per Minute) : Si tu fais plus de kills/min que d'habitude → score élevé
- **DPM** (Deaths Per Minute, inversé) : Mourir moins que d'habitude → score élevé  
- **APM** (Assists Per Minute) : Assister plus que d'habitude → score élevé
- **KDA** : Ratio global supérieur à ta moyenne → score élevé
- **Accuracy** : Précision au-dessus de ton niveau habituel → score élevé

---

## Formule v3-relative

### Métriques et pondérations

| Métrique | Poids | Direction |
|----------|-------|-----------|
| KPM (kills/min) | 30% | Plus haut = mieux |
| DPM (deaths/min) | 25% | Plus bas = mieux |
| APM (assists/min) | 15% | Plus haut = mieux |
| KDA | 20% | Plus haut = mieux |
| Accuracy | 10% | Plus haut = mieux |

### Calcul du percentile

Pour une métrique donnée, on calcule où se situe la valeur actuelle parmi les N matchs précédents :

$$
percentile = \frac{|\{x \in history : x < valeur\}|}{N} \times 100
$$

Pour les métriques inversées (DPM), on utilise le percentile inverse :

$$
percentile_{inverse} = 100 - \frac{|\{x \in history : x < valeur\}|}{N} \times 100
$$

### Score final

$$
score = \sum_{i} weight_i \times percentile_i
$$

---

## Paramètres configurables

Tous les paramètres sont centralisés dans `src/analysis/performance_config.py` :

```python
# Version du schéma de scoring
PERFORMANCE_SCORE_VERSION = "v3-relative"

# Minimum de matchs pour le calcul relatif
MIN_MATCHES_FOR_RELATIVE = 10

# Pondérations des composantes
RELATIVE_WEIGHTS = {
    "kpm": 0.30,      # Kills per minute
    "dpm": 0.25,      # Deaths per minute (inversé)
    "apm": 0.15,      # Assists per minute
    "kda": 0.20,      # (K + A) / D
    "accuracy": 0.10, # Précision
}

# Seuils d'interprétation
SCORE_THRESHOLDS = {
    "excellent": 75,
    "good": 60,
    "average": 45,
    "below_average": 30,
}
```

---

## Interprétation des scores

| Score | Interprétation | Signification |
|-------|----------------|---------------|
| ≥ 75 | 🌟 Excellent | Tu as surpassé tes performances habituelles |
| 60-74 | ✅ Bon | Au-dessus de ta moyenne |
| 45-59 | 📊 Moyen | Dans ta norme |
| 30-44 | 📉 En-dessous | Sous ta moyenne habituelle |
| < 30 | ⚠️ Mauvais | Performance inhabituelle (fatigue, distraction, warm-up) |

> **Important** : Un score de 50 signifie "performance typique pour toi", pas "performance médiocre".

---

## Stockage en base de données

Les scores sont calculés et stockés dans `MatchCache.performance_score` :

- **À l'import** : Le script delta sync calcule et stocke le score
- **Historique** : Le script `compute_historical_performance.py` recalcule tous les scores

### Pourquoi stocker le score ?

1. **Fige le contexte** : Le score reflète ton niveau *au moment du match*
2. **Évite la dérive** : En s'améliorant, ton ancien 70 resterait 70 (pas recalculé à 50)
3. **Performance** : Pas de recalcul à chaque affichage

---

## Migration depuis les anciennes versions

Pour recalculer tous les scores historiques avec l'algorithme relatif :

```bash
# Simulation (affiche ce qui serait fait)
python scripts/compute_historical_performance.py --dry-run

# Exécution réelle
python scripts/compute_historical_performance.py

# Forcer le recalcul même si les scores existent
python scripts/compute_historical_performance.py --force
```

Le script utilise une **approche rolling** : chaque match est comparé uniquement aux matchs **antérieurs**, pour refléter fidèlement le niveau du joueur à l'époque.

---

## Limites connues

1. **Premiers matchs** : Avec moins de 10 matchs historiques, le score peut être instable
2. **Changement de style** : Si tu changes radicalement de playstyle, les comparaisons sont moins pertinentes
3. **Modes différents** : Un match Firefight (PvE) comparé à des matchs PvP peut donner des résultats biaisés
4. **Sessions courtes** : 1-2 matchs = bruit statistique

---

## Évolutions possibles (roadmap)

- [ ] Segmentation par mode (PvP vs PvE, Arena vs BTB)
- [ ] Pondération dynamique selon la disponibilité des données
- [ ] Prise en compte de l'écart MMR (difficulté adverse)
- [ ] Score de confiance (intervalle selon la taille de l'historique)
- [ ] Comparaison inter-joueurs avec normalisation

---

## Historique des versions

| Version | Description |
|---------|-------------|
| v1 | Score absolu : K/D (30%) + Win rate (25%) + Accuracy (25%) + Match score (20%) |
| v2 | Score absolu modulaire : ajout objectifs, renormalisation si données manquantes |
| **v3-relative** | Score relatif aux performances personnelles, stocké en DB |
