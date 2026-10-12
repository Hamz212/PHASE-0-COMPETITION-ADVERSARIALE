# Phase 0 – Compétition de système de détection d'intrusion (IDS)

## Aperçu

Bienvenue à la phase 0 de la compétition d'apprentissage automatique adversarial!

Dans cette phase, votre objectif est de développer un **Feed-Forward Neural Network (FFNN)** capable de classifier le trafic réseau dans l'une des catégories suivantes :

* **0** – Légitime
* **1** – Malveillant

Vous entraînerez votre modèle à l'aide du jeu de données d'entraînement fourni, puis soumettrez votre modèle entraîné afin qu'il soit évalué sur un jeu de test caché.

---

# Objectifs d'apprentissage

À la fin de cette phase, vous serez en mesure de :

* Charger et prétraiter des jeux de données tabulaires
* Entraîner des réseaux de neurones avec PyTorch
* Effectuer l'optimisation des hyperparamètres
* Évaluer les performances d'un modèle
* Concevoir un système de détection d'intrusion (IDS)

---

# Fichiers fournis

```text
starter_submission/
├── scaledXTrain.csv
├── yTrain.csv
├── model.py
├── validateSubmission.py
├── requirements.txt
└── README.md
```

---

# Règles de la compétition

Vous **pouvez** modifier :

* L'architecture du réseau de neurones
* Le nombre de couches cachées
* Le nombre de neurones
* Les fonctions d'activation
* L'optimiseur
* Le taux d'apprentissage
* La taille des lots (*batch size*)
* Le nombre d'époques
* L'arrêt anticipé (*early stopping*)
* La régularisation
* Le *dropout*
* Le seuil de classification
* Toute autre technique d'entraînement applicable à un Feed-Forward Neural Network (FFNN)

Vous **ne pouvez pas** utiliser :

* Les réseaux de neurones convolutifs (CNN)
* Les réseaux de neurones récurrents (RNN)
* Les Transformers
* Les arbres de décision
* Les forêts aléatoires (*Random Forests*)
* Les arbres à gradient amplifié (*Gradient Boosted Trees*)
* Les machines à vecteurs de support (SVM)
* Toute architecture qui **n'est pas** un Feed-Forward Neural Network (MLP)

---

# Entraînement

Implémentez votre procédure d'entraînement dans :

```text
model.py
```

Le code de départ fourni contient :

* La classe `IDSModel`
* `forward()`
* `fit()`
* `predict()`

Vous pouvez modifier ces fonctions comme vous le souhaitez, à condition de respecter les règles de la compétition.

---

# Validation

Avant de soumettre votre solution, validez-la en exécutant :

```text
python validateSubmission.py
```

Le script de validation vérifie que :

* `submission.zip` existe
* `model.py` existe
* `model.pth` existe
* `IDSModel` peut être importé
* Les poids du modèle sont chargés correctement
* `predict()` s'exécute correctement
* Les prédictions ont la bonne dimension
* Les prédictions sont binaires (0 ou 1)

Une validation réussie **ne garantit pas** un score élevé; elle confirme uniquement que votre soumission peut être évaluée.

---

# Format de la soumission

Votre soumission finale **doit** être nommée :

```text
submission.zip
```

et contenir exactement :

```text
submission.zip
├── model.py
└── model.pth
```

Aucun fichier supplémentaire n'est requis.

---

# Évaluation

Votre soumission sera évaluée à l'aide d'un **jeu de test caché** qui **n'est pas** distribué aux étudiants.

Le processus d'évaluation est le suivant :

```text
submission.zip
        │
        ▼
Extraction de la soumission
        │
        ▼
Importation de model.py
        │
        ▼
Chargement de model.pth
        │
        ▼
Prédiction sur le jeu de test caché
        │
        ▼
Calcul des métriques d'évaluation
```

Le script d'évaluation de l'organisateur suit ce processus afin de charger votre modèle et de calculer les résultats finaux.

---

# Évaluation des performances

Les métriques suivantes seront rapportées :

* Exactitude (*Accuracy*)
* Précision (*Precision*)
* Rappel (*Recall*)
* Score F1

La métrique principale de la compétition est :

> **Score F1**

---

# Conseils

* Commencez par le modèle fourni avant d'expérimenter d'autres approches.
* Modifiez un seul élément à la fois.
* Surveillez les performances du modèle pendant l'entraînement.
* Privilégiez une implémentation simple avant d'essayer des approches plus avancées.
* Validez toujours votre soumission avant de la remettre.

Bonne chance et amusez-vous!
