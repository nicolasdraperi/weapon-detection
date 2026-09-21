"""
training.py : Pipeline complète de détection d'armes YOLOv8
==============================================================
Mémoire : L'IA dans l'investigation criminelle

Ce script fait TOUT dans l'ordre :
    0. Téléchargement du dataset (Roboflow) si absent localement
    1. Vérification de l'environnement (GPU, librairies)
    2. Vérification du dataset brut
    3. Re-split stratifié 70 / 20 / 10 (équilibre des classes)
    4. Entraînement YOLOv8 (fine-tuning transfer learning)
    5. Évaluation sur le jeu de test (mAP, précision, rappel)
    6. Génération des graphiques pour le mémoire
    7. Test de détection sur une image ou vidéo
    8. Export du rapport JSON final

Usage :
    python training.py                        # pipeline complète
    python training.py --no-collect           # skip téléchargement
    python training.py --skip-resplit         # skip re-split
    python training.py --skip-train           # skip entraînement
    python training.py --detect chemin/img    # détection sur fichier
    python training.py --weights best.pt      # spécifier un modèle

Prérequis :
    pip install ultralytics roboflow opencv-python python-dotenv iterative-stratification matplotlib numpy

Clé API Roboflow dans un fichier .env à la racine :
    ROBOFLOW_API_KEY=ta_cle_api
"""

import os
import sys
import csv
import json
import time
import shutil
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

# Chargement de la clé API depuis .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv optionnel si la clé est déjà dans l'environnement

# CONFIGURATION CENTRALE
# Roboflow
ROBOFLOW_WORKSPACE = "test-7awfy"
ROBOFLOW_PROJECT   = "weapon-detection-f1lih"
ROBOFLOW_VERSION   = 1

# Dossiers
RAW_DATASET_DIR      = Path("weapon-detection-raw")       # dataset tel que téléchargé
RESPLIT_DATASET_DIR  = Path("weapon-detection-resplit")   # dataset après re-split
RESULTS_DIR          = Path("results")                     # sorties entraînement
PLOTS_DIR            = RESULTS_DIR / "plots"
RUN_NAME             = "weapon_detector_v1"

# Dataset
CLASSES = ['Grenade', 'Knife', 'Missile', 'Pistol', 'Rifle']
NC      = len(CLASSES)

# Proportions train / valid / test
RATIO_TRAIN = 0.70
RATIO_VALID = 0.20
RATIO_TEST  = 0.10
SPLIT_SEED  = 42   # reproductibilité

# Entraînement
TRAIN_CONFIG = {
    "model":         "yolov8s.pt",  # n=rapide/CPU  s=équilibré  m=précis/GPU
    "epochs":        100,
    "imgsz":         640,
    "batch":         16,
    "patience":      20,            # early stopping
    "conf_detect":   0.5,           # seuil pour la DÉTECTION (app + tests visuels)
    "conf_eval":     0.001,         # seuil pour le CALCUL des métriques (courbe P-R complète)
    "iou":           0.45,
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Couleurs BGR par classe (pour OpenCV)
CLASS_COLORS_BGR = {
    "Grenade": (0,   0, 220),
    "Knife":   (0, 140, 255),
    "Missile": (200, 80,   0),
    "Pistol":  (200,  0,  50),
    "Rifle":   (0, 180,  40),
}


# HELPERS D'AFFICHAGE
def header(n, title):
    print(f"\n{'═'*62}")
    print(f"  ÉTAPE {n} : {title}")
    print(f"{'═'*62}")

def ok(msg):    print(f"  [✓] {msg}")
def warn(msg):  print(f"  [!] {msg}")
def info(msg):  print(f"      {msg}")

def err(msg):
    print(f"\n  [✗] ERREUR : {msg}\n")
    sys.exit(1)



# ÉTAPE 0 : TÉLÉCHARGEMENT DU DATASET
def _find_existing_dataset():
    """Cherche un dataset déjà présent localement (data.yaml + train/)."""
    candidates = [
        RAW_DATASET_DIR,
        RESPLIT_DATASET_DIR,
        Path(f"{ROBOFLOW_PROJECT}-{ROBOFLOW_VERSION}"),
        Path("weapon detection.v1i.yolov8"),
    ]
    for c in candidates:
        if (c / "data.yaml").exists() and (c / "train").exists():
            return c
    # Recherche élargie
    for yml in Path(".").glob("*/data.yaml"):
        if (yml.parent / "train").exists():
            return yml.parent
    return None


def step0_collect(force_skip: bool):
    header(0, "Téléchargement du dataset")
    global RAW_DATASET_DIR

    found = _find_existing_dataset()
    if found:
        RAW_DATASET_DIR = found
        ok(f"Dataset trouvé localement → {RAW_DATASET_DIR}  (téléchargement ignoré)")
        return

    if force_skip:
        err("--no-collect activé mais aucun dataset trouvé.\n"
            "      Lance sans --no-collect pour le télécharger.")

    try:
        from roboflow import Roboflow
    except ImportError:
        err("Package manquant : pip install roboflow")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        err("Clé API manquante.\n"
            "      Crée un fichier .env contenant : ROBOFLOW_API_KEY=ta_cle\n"
            "      ou exporte la variable : export ROBOFLOW_API_KEY=ta_cle")

    info(f"Connexion à Roboflow : {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=str(RAW_DATASET_DIR))

    RAW_DATASET_DIR = Path(dataset.location)
    ok(f"Dataset téléchargé → {RAW_DATASET_DIR}")



# ÉTAPE 1 : VÉRIFICATION DE L'ENVIRONNEMENT
def step1_environment():
    header(1, "Vérification de l'environnement")

    v = sys.version_info
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

    try:
        import torch
        ok(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"GPU : {gpu}  ({mem:.1f} GB)  →  entraînement rapide")
        else:
            warn("Pas de GPU détecté, entraînement sur CPU (lent)")
    except ImportError:
        err("PyTorch manquant : pip install ultralytics")

    try:
        import ultralytics
        ok(f"Ultralytics {ultralytics.__version__}")
    except ImportError:
        err("Ultralytics manquant : pip install ultralytics")

    try:
        import cv2
        ok(f"OpenCV {cv2.__version__}")
    except ImportError:
        err("OpenCV manquant : pip install opencv-python")

    try:
        import matplotlib
        ok(f"Matplotlib {matplotlib.__version__}")
    except ImportError:
        warn("Matplotlib absent, les graphiques ne seront pas générés")



# ÉTAPE 2 : VÉRIFICATION DU DATASET BRUT
def step2_check_raw():
    header(2, "Vérification du dataset brut")

    yaml_path = RAW_DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        err(f"data.yaml introuvable dans {RAW_DATASET_DIR}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    ok(f"{cfg['nc']} classes : {cfg['names']}")

    total = 0
    for split in ("train", "valid", "test"):
        img_dir = RAW_DATASET_DIR / split / "images"
        lbl_dir = RAW_DATASET_DIR / split / "labels"
        if not img_dir.exists():
            warn(f"Dossier manquant : {img_dir}")
            continue
        imgs   = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        ok(f"{split:6s} → {len(imgs):5d} images  |  {len(labels):5d} labels")
        total += len(imgs)

    ok(f"Total brut : {total} images")



# ÉTAPE 3 : RE-SPLIT STRATIFIÉ 70 / 20 / 10


def _find_image(label_path: Path, images_dir: Path):
    for ext in IMG_EXTS:
        c = images_dir / f"{label_path.stem}{ext}"
        if c.exists():
            return c
    return None


def _class_vector(label_path: Path):
    vec = np.zeros(NC, dtype=int)
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            idx = int(parts[0])
            if 0 <= idx < NC:
                vec[idx] = 1
    return vec


def step3_resplit(force_skip: bool):
    header(3, "Re-split stratifié 70 / 20 / 10")

    if force_skip:
        warn("--skip-resplit activé, re-split ignoré")
        if not (RESPLIT_DATASET_DIR / "data.yaml").exists():
            warn("Dataset re-splitté absent, utilisation du dataset brut")
            return RAW_DATASET_DIR
        ok(f"Dataset re-splitté existant → {RESPLIT_DATASET_DIR}")
        return RESPLIT_DATASET_DIR

    if RESPLIT_DATASET_DIR.exists():
        warn(f"{RESPLIT_DATASET_DIR} existe déjà, re-split ignoré")
        ok(f"Pour forcer le re-split : supprime le dossier {RESPLIT_DATASET_DIR}")
        return RESPLIT_DATASET_DIR

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        warn("iterative-stratification absent, re-split ignoré (split Roboflow conservé)")
        warn("Pour activer : pip install iterative-stratification")
        return RAW_DATASET_DIR

    # Collecte de toutes les images annotées
    info("Collecte des images…")
    samples = []
    for split in ("train", "valid", "test"):
        lbl_dir = RAW_DATASET_DIR / split / "labels"
        img_dir = RAW_DATASET_DIR / split / "images"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            img = _find_image(lbl, img_dir)
            if img is None:
                continue
            vec = _class_vector(lbl)
            if vec.sum() == 0:
                continue
            samples.append((lbl, img, vec))

    info(f"{len(samples)} images annotées collectées")

    # Stratification multi-label
    X = np.arange(len(samples)).reshape(-1, 1)
    Y = np.array([s[2] for s in samples])

    # A : train (70%) vs reste (30%)
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(RATIO_VALID + RATIO_TEST), random_state=SPLIT_SEED
    )
    train_idx, rest_idx = next(msss1.split(X, Y))

    # B : valid (20%) vs test (10%) dans le reste
    test_share = RATIO_TEST / (RATIO_VALID + RATIO_TEST)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_share, random_state=SPLIT_SEED
    )
    valid_local, test_local = next(msss2.split(X[rest_idx], Y[rest_idx]))
    valid_idx = rest_idx[valid_local]
    test_idx  = rest_idx[test_local]

    splits_idx = {
        "train": train_idx,
        "valid": valid_idx,
        "test":  test_idx,
    }

    # Copie des fichiers
    for split, idxs in splits_idx.items():
        (RESPLIT_DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (RESPLIT_DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in idxs:
            lbl, img, _ = samples[i]
            shutil.copy2(img, RESPLIT_DATASET_DIR / split / "images" / img.name)
            shutil.copy2(lbl, RESPLIT_DATASET_DIR / split / "labels" / lbl.name)
        ok(f"{split:6s} → {len(idxs)} images copiées")

    # data.yaml
    yaml_content = {
        "train": str((RESPLIT_DATASET_DIR / "train" / "images").resolve()),
        "val":   str((RESPLIT_DATASET_DIR / "valid" / "images").resolve()),
        "test":  str((RESPLIT_DATASET_DIR / "test"  / "images").resolve()),
        "nc":    NC,
        "names": CLASSES,
    }
    with open(RESPLIT_DATASET_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, allow_unicode=True)

    # Vérification de l'équilibre
    print(f"\n  Répartition des instances par classe :")
    print(f"  {'Classe':10s} | {'train':>6s} | {'valid':>6s} | {'test':>6s}")
    print("  " + "─" * 38)
    per = {s: Counter() for s in ("train", "valid", "test")}
    for split, idxs in splits_idx.items():
        for i in idxs:
            lbl = samples[i][0]
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    per[split][int(parts[0])] += 1
    for i, name in enumerate(CLASSES):
        print(f"  {name:10s} | {per['train'][i]:6d} | {per['valid'][i]:6d} | {per['test'][i]:6d}")

    ok(f"\n  Dataset re-splitté → {RESPLIT_DATASET_DIR.resolve()}")
    return RESPLIT_DATASET_DIR



# ÉTAPE 4 : PRÉPARATION DU data.yaml POUR L'ENTRAÎNEMENT


def step4_prepare_yaml(dataset_dir: Path):
    header(4, "Préparation du data.yaml")

    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        err(f"data.yaml introuvable dans {dataset_dir}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Chemins absolus (évite les problèmes selon le répertoire de lancement)
    cfg["train"] = str((dataset_dir / "train" / "images").resolve())
    cfg["val"]   = str((dataset_dir / "valid" / "images").resolve())
    cfg["test"]  = str((dataset_dir / "test"  / "images").resolve())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fixed_yaml = RESULTS_DIR / "data_training.yaml"
    with open(fixed_yaml, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    ok(f"data.yaml prêt → {fixed_yaml}")
    ok(f"Classes : {cfg['names']}")
    for split_key, label in [("train","train"), ("val","valid"), ("test","test")]:
        img_dir = Path(cfg[split_key])
        count = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS) if img_dir.exists() else 0
        ok(f"{label:6s} → {count} images")

    return str(fixed_yaml)



#  ÉTAPE 5 : ENTRAÎNEMENT


def step5_train(data_yaml: str):
    header(5, "Entraînement YOLOv8")

    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    cfg    = TRAIN_CONFIG

    print(f"\n  Modèle      : {cfg['model']}  (fine-tuning depuis COCO)")
    print(f"  Epochs      : {cfg['epochs']}  (early stop après {cfg['patience']} sans amélioration)")
    print(f"  Image size  : {cfg['imgsz']} px")
    print(f"  Batch size  : {cfg['batch']}")
    print(f"  Device      : {'GPU' if device == 0 else 'CPU'}\n")

    model = YOLO(cfg["model"])
    t0    = time.time()

    model.train(
        data          = data_yaml,
        epochs        = cfg["epochs"],
        imgsz         = cfg["imgsz"],
        batch         = cfg["batch"],
        device        = device,
        workers       = 2,
        project       = str(RESULTS_DIR),
        name          = RUN_NAME,
        patience      = cfg["patience"],
        exist_ok      = True,
        # Augmentations (simulation vidéosurveillance)
        hsv_h         = 0.015,   # variation de teinte (caméras IR, nuit)
        hsv_s         = 0.7,     # variation de saturation
        hsv_v         = 0.4,     # variation de luminosité
        fliplr        = 0.5,     # flip horizontal 50 %
        mosaic        = 1.0,     # mosaïque 4 images → petits objets
        mixup         = 0.1,     # superposition de 2 images
        # Hyperparamètres d'optimisation
        lr0           = 0.01,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 3,
        plots         = True,
        save          = True,
        verbose       = True,
    )

    elapsed      = time.time() - t0
    best_weights = Path(model.trainer.best)

    if not best_weights.exists():
        err("best.pt introuvable après entraînement")

    ok(f"Entraînement terminé en {elapsed/60:.1f} min")
    ok(f"Meilleur modèle → {best_weights}")
    return str(best_weights)



#  ÉTAPE 6 : ÉVALUATION


def step6_evaluate(weights_path: str, data_yaml: str):
    header(6, "Évaluation sur le jeu de test")

    from ultralytics import YOLO
    import torch

    device  = 0 if torch.cuda.is_available() else "cpu"
    model   = YOLO(weights_path)

    # conf=0.001 → calcul de la courbe Précision-Rappel complète (standard mAP)
    metrics = model.val(
        data      = data_yaml,
        split     = "test",
        conf      = TRAIN_CONFIG["conf_eval"],
        iou       = TRAIN_CONFIG["iou"],
        device    = device,
        plots     = True,
        save_json = True,
    )

    results = {
        "mAP_50":    round(float(metrics.box.map50), 4),
        "mAP_50_95": round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),    4),
        "recall":    round(float(metrics.box.mr),    4),
    }

    print(f"\n  {'Métrique':<20} {'Valeur':>8}  {'%':>7}")
    print("  " + "─" * 40)
    print(f"  {'mAP@0.5':<20} {results['mAP_50']:>8.4f}  {results['mAP_50']*100:>6.1f}%")
    print(f"  {'mAP@0.5:0.95':<20} {results['mAP_50_95']:>8.4f}  {results['mAP_50_95']*100:>6.1f}%")
    print(f"  {'Précision':<20} {results['precision']:>8.4f}  {results['precision']*100:>6.1f}%")
    print(f"  {'Rappel':<20} {results['recall']:>8.4f}  {results['recall']*100:>6.1f}%")

    return results, metrics



#  ÉTAPE 7 : GRAPHIQUES POUR LE MÉMOIRE


def step7_plots(results: dict, metrics):
    header(7, "Génération des graphiques")

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        warn("matplotlib ou pandas absent, graphiques ignorés")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 7a : Courbes d'entraînement
    csv_path = RESULTS_DIR / RUN_NAME / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Courbes d'entraînement : YOLOv8 Weapon Detection", fontweight="bold")

        axes[0].plot(df["epoch"], df["train/box_loss"], label="box loss train", color="#1a56db")
        axes[0].plot(df["epoch"], df["val/box_loss"],   label="box loss val",   color="#1a56db", linestyle="--")
        axes[0].plot(df["epoch"], df["train/cls_loss"], label="cls loss train", color="#c81e1e")
        axes[0].plot(df["epoch"], df["val/cls_loss"],   label="cls loss val",   color="#c81e1e", linestyle="--")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Époques")
        axes[0].legend(); axes[0].grid(alpha=0.4)

        axes[1].plot(df["epoch"], df["metrics/mAP50(B)"],     label="mAP@0.5",      color="#057a55")
        axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"],  label="mAP@0.5:0.95", color="#057a55", linestyle="--")
        axes[1].plot(df["epoch"], df["metrics/precision(B)"], label="Précision",     color="#1a56db")
        axes[1].plot(df["epoch"], df["metrics/recall(B)"],    label="Rappel",        color="#c81e1e")
        axes[1].set_title("Métriques"); axes[1].set_xlabel("Époques")
        axes[1].legend(); axes[1].grid(alpha=0.4)

        plt.tight_layout()
        out = PLOTS_DIR / "courbes_entrainement.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        ok(f"Courbes d'entraînement → {out}")
    else:
        warn(f"results.csv introuvable ({csv_path}), courbes ignorées")

    # 7b : Performance par classe
    maps_per_class = list(getattr(metrics.box, "maps", []))
    if maps_per_class and len(maps_per_class) == NC:
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(CLASSES, maps_per_class, color="#1a56db", alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mAP@0.5:0.95")
        ax.set_title("Performance par classe d'arme : jeu de test", fontweight="bold")
        ax.grid(axis="y", alpha=0.4)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", fontsize=10)
        plt.tight_layout()
        out = PLOTS_DIR / "performance_par_classe.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        ok(f"Performance par classe → {out}")

    # 7c : Comparaison IA vs analyse humaine
    categories   = ["Précision", "Rappel", "mAP@0.5"]
    ia_values    = [results["precision"], results["recall"], results["mAP_50"]]
    human_values = [0.78, 0.72, 0.65]   # valeurs de référence littérature

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Comparaison : Système IA vs Analyse humaine", fontweight="bold")
    b1 = ax.bar(x - 0.175, ia_values,    0.35, label="YOLOv8 (ce projet)", color="#1a56db", alpha=0.85)
    b2 = ax.bar(x + 0.175, human_values, 0.35, label="Analyste humain",    color="#6b7280", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.15); ax.legend(); ax.grid(axis="y", alpha=0.4)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{b.get_height():.2f}", ha="center", fontsize=10)
    ax.text(0.99, 0.02, "* valeurs humaines : référence littérature (Gruosso et al., 2021)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#6b7280")
    plt.tight_layout()
    out = PLOTS_DIR / "ia_vs_humain.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    ok(f"Comparaison IA vs humain → {out}")

    print(f"\n  Tous les graphiques sont dans : {PLOTS_DIR.resolve()}")



#  ÉTAPE 8 : TEST DE DÉTECTION SUR FICHIER


def step8_detect(weights_path: str, source: str):
    header(8, f"Détection sur : {source}")

    import cv2
    from ultralytics import YOLO

    model       = YOLO(weights_path)
    source_path = Path(source)
    is_image    = source_path.suffix.lower() in IMG_EXTS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def annotate_frame(frame, results):
        dets = []
        h, w = frame.shape[:2]
        for r in results:
            for box in (r.boxes or []):
                conf = float(box.conf[0])
                if conf < TRAIN_CONFIG["conf_detect"]:
                    continue
                cls_name = r.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = CLASS_COLORS_BGR.get(cls_name, (0, 0, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name}  {conf:.0%}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1-th-bl-6), (x1+tw+6, y1), color, -1)
                cv2.putText(frame, label, (x1+3, y1-bl-2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                dets.append({"class": cls_name, "conf": round(conf, 3), "bbox": [x1,y1,x2,y2]})
        if dets:
            cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 180), -1)
            cv2.putText(frame, f"  ALERTE : {len(dets)} ARME(S) DETECTEE(S)",
                        (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return frame, dets

    if is_image:
        img = cv2.imread(source)
        if img is None:
            err(f"Image introuvable : {source}")
        res          = model(img, verbose=False)
        ann, dets    = annotate_frame(img.copy(), res)
        out          = RESULTS_DIR / f"detected_{source_path.name}"
        cv2.imwrite(str(out), ann)
        ok(f"Image annotée → {out}")
        if dets:
            for d in dets:
                info(f"→ {d['class']:10s}  conf={d['conf']:.0%}  bbox={d['bbox']}")
        else:
            info("Aucune arme détectée")

    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            err(f"Impossible d'ouvrir : {source}")

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        out_path = RESULTS_DIR / f"detected_{source_path.stem}.mp4"
        writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        all_dets = []; frame_id = 0; t0 = time.time()
        info("Traitement vidéo en cours… (Ctrl+C pour arrêter)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res          = model(frame, verbose=False)
            ann, dets    = annotate_frame(frame.copy(), res)
            all_dets.extend(dets)
            writer.write(ann)
            frame_id += 1
            if frame_id % 100 == 0:
                fps_proc = frame_id / (time.time() - t0)
                info(f"Frame {frame_id}  |  {fps_proc:.1f} FPS  |  {len(all_dets)} détections")

        cap.release(); writer.release()
        ok(f"Vidéo annotée → {out_path}")
        ok(f"{frame_id} frames analysées  |  {len(all_dets)} détections au total")

        # Export CSV chronologie
        if all_dets:
            csv_out = RESULTS_DIR / f"chronologie_{source_path.stem}.csv"
            with open(csv_out, "w", newline="", encoding="utf-8") as f:
                w_csv = csv.writer(f, delimiter=";")
                w_csv.writerow(["Frame", "Classe", "Confiance", "BBox"])
                for d in all_dets:
                    w_csv.writerow([d.get("frame",""), d["class"], f"{d['conf']:.2%}", d["bbox"]])
            ok(f"Chronologie CSV → {csv_out}")



#  ÉTAPE 9 : RAPPORT JSON FINAL


def step9_report(results: dict, weights_path: str, dataset_dir: Path):
    header(9, "Rapport final")

    report = {
        "generated_at":   datetime.now().isoformat(),
        "model_weights":  weights_path,
        "dataset":        str(dataset_dir.resolve()),
        "classes":        CLASSES,
        "split_ratios":   {"train": RATIO_TRAIN, "valid": RATIO_VALID, "test": RATIO_TEST},
        "train_config":   TRAIN_CONFIG,
        "metrics":        results,
        "interpretation": {
            "mAP_50":    "Précision moyenne à IoU=0.5, métrique principale",
            "mAP_50_95": "Métrique stricte standard COCO (plus exigeante)",
            "precision": "Taux de vraies alertes parmi toutes les alertes émises (évite les faux positifs)",
            "recall":    "Taux d'armes détectées parmi toutes les armes présentes (évite les faux négatifs)",
        },
        "plots": [str(p) for p in PLOTS_DIR.glob("*.png")] if PLOTS_DIR.exists() else [],
    }

    out = RESULTS_DIR / f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    ok(f"Rapport JSON → {out}")

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  RÉSUMÉ FINAL                                        │
  ├──────────────────────────────────────────────────────┤
  │  Classes  : {', '.join(CLASSES)}
  │  Dataset  : {str(dataset_dir)}
  │  Modèle   : {TRAIN_CONFIG['model']}  ({TRAIN_CONFIG['epochs']} epochs max)
  ├──────────────────────────────────────────────────────┤
  │  mAP@0.5       : {results.get('mAP_50', '?')}
  │  mAP@0.5:0.95  : {results.get('mAP_50_95', '?')}
  │  Précision     : {results.get('precision', '?')}
  │  Rappel        : {results.get('recall', '?')}
  ├──────────────────────────────────────────────────────┤
  │  Graphiques : {str(PLOTS_DIR)}
  │  Rapport    : {str(out)}
  └──────────────────────────────────────────────────────┘
""")



# MAIN


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complète de détection d'armes YOLOv8",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--no-collect",   action="store_true", help="Ne pas télécharger le dataset")
    parser.add_argument("--skip-resplit", action="store_true", help="Ne pas re-splitter le dataset")
    parser.add_argument("--skip-train",   action="store_true", help="Ne pas entraîner (utilise best.pt existant)")
    parser.add_argument("--detect",       type=str, default=None, help="Image ou vidéo à analyser")
    parser.add_argument("--weights",      type=str, default=None, help="Chemin vers best.pt")
    args = parser.parse_args()

    print("\n" + "█"*62)
    print("  training.py : Détection d'armes par IA")
    print("  Mémoire : L'IA dans l'investigation criminelle")
    print("█"*62)

    # Étape 0 : Téléchargement
    step0_collect(force_skip=args.no_collect)

    # Étape 1 : Environnement
    step1_environment()

    # Étape 2 : Vérification dataset brut
    step2_check_raw()

    # Étape 3 : Re-split
    dataset_dir = step3_resplit(force_skip=args.skip_resplit)

    # Étape 4 : data.yaml
    data_yaml = step4_prepare_yaml(dataset_dir)

    # Mode détection directe (--detect)
    if args.detect:
        weights = args.weights or str(RESULTS_DIR / RUN_NAME / "weights" / "best.pt")
        if not Path(weights).exists():
            err(f"Modèle introuvable : {weights}\nLance d'abord training.py sans --detect")
        step8_detect(weights, args.detect)
        return

    # Étape 5 : Entraînement
    if args.skip_train:
        weights = args.weights or str(RESULTS_DIR / RUN_NAME / "weights" / "best.pt")
        if not Path(weights).exists():
            err(f"best.pt introuvable : {weights}")
        warn("Entraînement ignoré, modèle existant utilisé")
    else:
        weights = step5_train(data_yaml)

    # Étape 6 : Évaluation
    results, metrics = step6_evaluate(weights, data_yaml)

    # Étape 7 : Graphiques
    step7_plots(results, metrics)

    # Étape 9 : Rapport
    step9_report(results, weights, dataset_dir)

    print("\n  ✓ Pipeline terminée avec succès !\n")


if __name__ == "__main__":
    main()
