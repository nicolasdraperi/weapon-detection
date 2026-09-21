# Détection d'armes par IA — YOLOv8

> Mémoire : *L'intelligence artificielle dans l'investigation criminelle*  
> Système de détection automatique d'armes sur images et flux vidéo de vidéosurveillance

---

## Présentation

Ce projet implémente un système de détection d'armes en temps réel basé sur **YOLOv8** (You Only Look Once v8), entraîné par transfer learning sur un dataset annoté de plus de 9 600 images.

**Classes détectées :** Grenade · Knife · Missile · Pistol · Rifle

**Performances obtenues (jeu de test) :**

| Métrique | Valeur |
|---|---|
| mAP@0.5 | 83.2 % |
| mAP@0.5:0.95 | 59.5 % |
| Précision | 85.8 % |
| Rappel | 75.7 % |

---

## Structure du projet

```
.
├── training.py                        ← Pipeline complète (téléchargement → entraînement → évaluation)
├── app.py                             ← Interface web Gradio (image / vidéo / webcam)
├── requirements.txt                   ← Dépendances Python
├── .env                               ← Clé API Roboflow (non versionné)
├── .gitignore
│
├── weapon-detection-raw/              ← Dataset brut téléchargé depuis Roboflow
│   ├── train/images/ + labels/
│   ├── valid/images/ + labels/
│   ├── test/images/  + labels/
│   └── data.yaml
│
├── weapon-detection-resplit/          ← Dataset après re-split stratifié 70/20/10
│   ├── train/ · valid/ · test/
│   └── data.yaml
│
└── results/                           ← Sorties générées automatiquement
    ├── data_training.yaml
    ├── weapon_detector_v1/
    │   ├── weights/
    │   │   ├── best.pt                ← Meilleur modèle (à utiliser)
    │   │   └── last.pt
    │   ├── results.csv
    │   ├── confusion_matrix.png
    │   ├── BoxPR_curve.png
    │   └── ...
    ├── plots/
    │   ├── courbes_entrainement.png   ← Pour le mémoire (section 3.2)
    │   ├── performance_par_classe.png ← Pour le mémoire (section 3.2)
    │   └── ia_vs_humain.png           ← Pour le mémoire (section 3.2)
    └── rapport_YYYYMMDD_HHMMSS.json  ← Rapport final exportable
```

---

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/ton-user/weapon-detection.git
cd weapon-detection
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **GPU Nvidia recommandé.** Si PyTorch ne détecte pas ton GPU :
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Configurer la clé API Roboflow

Crée un fichier `.env` à la racine :

```
ROBOFLOW_API_KEY=ta_cle_api_ici
```

Ta clé API est disponible sur [app.roboflow.com](https://app.roboflow.com) → Settings → API Keys.

---

## Utilisation

### Pipeline complète (de A à Z)

```bash
python training.py
```

Exécute dans l'ordre :
1. Téléchargement du dataset depuis Roboflow
2. Vérification de l'environnement
3. Vérification du dataset brut
4. Re-split stratifié 70 / 20 / 10
5. Entraînement YOLOv8 (fine-tuning)
6. Évaluation sur le jeu de test
7. Génération des graphiques
8. Export du rapport JSON

---

### Options disponibles

```bash
# Dataset déjà téléchargé localement → skip le téléchargement
python training.py --no-collect

# Dataset déjà re-splitté → skip le re-split
python training.py --skip-resplit

# Modèle déjà entraîné → skip l'entraînement (évaluation + graphiques uniquement)
python training.py --skip-train

# Tester le modèle sur une image
python training.py --detect image.jpg

# Tester le modèle sur une vidéo
python training.py --detect video.mp4

# Spécifier un modèle particulier
python training.py --skip-train --weights results/weapon_detector_v1/weights/best.pt
```

---

### Interface web (app.py)

```bash
python app.py
```

Ouvre ensuite [http://127.0.0.1:7860](http://127.0.0.1:7860) dans ton navigateur.

L'interface propose trois modes :
- **Image** — dépose une photo, obtiens le résultat annoté instantanément
- **Vidéo** — analyse frame par frame, vidéo annotée téléchargeable + export CSV chronologie
- **Webcam** — détection en direct depuis ta caméra (~10 fps)

> `app.py` nécessite uniquement `best.pt`. Vérifie que le chemin `WEIGHTS` en haut du fichier pointe vers ton modèle entraîné.

---

## Dataset

**Source :** [Roboflow Universe — Weapon Detection](https://universe.roboflow.com/test-7awfy/weapon-detection-f1lih)  
**Licence :** CC BY 4.0  
**Volume :** 9 630 images annotées au format YOLOv8

### Re-split stratifié

Le dataset Roboflow d'origine présente un déséquilibre : la classe **Missile** est absente des jeux de validation et de test. `training.py` applique automatiquement un re-split **stratifié multi-label** (bibliothèque `iterative-stratification`) qui garantit la présence de chaque classe dans les trois splits.

| Split | Ratio | Rôle |
|---|---|---|
| train | 70 % | Apprentissage du modèle |
| valid | 20 % | Ajustement des hyperparamètres |
| test  | 10 % | Évaluation finale (données jamais vues) |

---

## Modèle

### Architecture

**YOLOv8s** (Small) — 11,1 M paramètres — 28,4 GFLOPs

| Composant | Détail |
|---|---|
| Backbone | CSPNet (extraction de features) |
| Neck | FPN + PAN (fusion multi-échelle) |
| Head | Détection découplée (classification + localisation) |
| Entraînement | Transfer learning depuis poids COCO |

### Hyperparamètres principaux

| Paramètre | Valeur | Justification |
|---|---|---|
| `epochs` | 100 | Avec early stopping (patience=20) |
| `imgsz` | 640 | Standard YOLOv8, bon compromis vitesse/précision |
| `batch` | 16 | Adapté à une GPU 6 GB |
| `lr0` | 0.01 | Learning rate initial |
| `mosaic` | 1.0 | Robustesse sur petits objets |
| `mixup` | 0.1 | Meilleure généralisation |

### Augmentations de données

Les augmentations suivantes améliorent la robustesse sur les images de vidéosurveillance (faible luminosité, angles variés, qualité variable) :

- **HSV** — variations de teinte, saturation et luminosité (caméras IR, nuit)
- **Flip horizontal** — 50 % des images retournées
- **Mosaic** — assemblage de 4 images (détection de petits objets)
- **Mixup** — superposition de deux images (meilleure généralisation)

---

## Résultats

### Métriques globales

| Métrique | Valeur | Interprétation |
|---|---|---|
| mAP@0.5 | 83.2 % | Précision moyenne à IoU=0.5 — métrique principale |
| mAP@0.5:0.95 | 59.5 % | Métrique stricte standard COCO |
| Précision | 85.8 % | Taux de vraies alertes parmi toutes les alertes |
| Rappel | 75.7 % | Taux d'armes réellement détectées |

### Métriques par classe

| Classe | Précision | Rappel | mAP@0.5 |
|---|---|---|---|
| Grenade | 0.898 | 0.837 | 0.896 |
| Pistol | 0.867 | 0.751 | 0.837 |
| Rifle | 0.872 | 0.759 | 0.838 |
| Knife | 0.795 | 0.682 | 0.758 |

### Vitesse de traitement

| Hardware | FPS | Temps par image |
|---|---|---|
| RTX 3060 Laptop (6 GB) | ~60 fps | ~4 ms |
| CPU (i7) | ~3 fps | ~350 ms |

---

## Limites connues

- **Biais de pose** — une arme tenue dans un angle inhabituel (au-dessus de la tête, partiellement cachée) peut ne pas être détectée
- **Classe Knife** — moins bien détectée car visuellement plus petite et moins distinctive
- **Dépendance au dataset** — les armes non représentées dans les données d'entraînement ne sont pas reconnues
- **Faux positifs** — certains objets (téléphones noirs, outils allongés) peuvent être confondus avec des armes à faible seuil de confiance
- **Conditions nocturnes** — les performances se dégradent sur des images très sombres ou à fort bruit

---

## Correspondance avec le mémoire

| Fichier / Sortie | Section du mémoire |
|---|---|
| `training.py` (architecture, config) | 2.1 — Conception du système |
| `training.py` (augmentations, hyperparamètres) | 2.2 — Fonctionnement technique |
| `training.py` (seuil confiance, rapport JSON) | 2.3 — Contrôle humain et éthique |
| `app.py` + détection vidéo surveillance | 3.1 — Cas d'utilisation concret |
| `plots/` + tableau métriques | 3.2 — Analyse des résultats |
| Section Limites connues ci-dessus | 3.3 — Limites observées |

---

## Références

- Redmon, J. & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement*
- Jocher, G. et al. (2023). *Ultralytics YOLOv8* — https://github.com/ultralytics/ultralytics
- Dataset : Roboflow Universe — Weapon Detection (CC BY 4.0)
- Soh, J. W. et al. (2019). *Real-time Weapon Detection Using Deep Learning*

---

## Licence

Ce projet est réalisé dans le cadre d'un mémoire académique.  
Le dataset est soumis à la licence **CC BY 4.0** (attribution requise).
