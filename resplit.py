"""
Re-split stratifié du dataset de détection d'armes.

Objectif :
    Le dataset d'origine (weapon-detection-1) a une répartition déséquilibrée :
    la classe Missile est absente de valid/test, et Grenade/Knife y sont
    sous-représentées. Ce script reconstruit un découpage propre 70/20/10 en
    respectant au mieux la proportion de CHAQUE classe dans chaque split
    (stratification multi-label).

Principe :
    - On regroupe toutes les images des 3 splits d'origine.
    - On les redistribue avec une stratification multi-label (chaque image peut
      contenir plusieurs classes → on respecte les proportions de toutes à la fois).
    - On COPIE (jamais on ne déplace) vers un nouveau dossier weapon-detection-resplit.
    - L'original reste totalement intact.

Prérequis :
    pip install iterative-stratification numpy

Usage :
    python resplit.py
"""

import shutil
import numpy as np
from pathlib import Path
from collections import Counter

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    raise SystemExit(
        "Bibliothèque manquante. Lance d'abord :\n"
        "    pip install iterative-stratification numpy"
    )


# Configuration

SRC_DIR  = Path("weapon-detection-1")          # dataset d'origine (lecture seule)
DST_DIR  = Path("weapon-detection-resplit")    # nouveau dataset (créé par ce script)
CLASSES  = ['Grenade', 'Knife', 'Missile', 'Pistol', 'Rifle']
NC       = len(CLASSES)

# Proportions 70 / 20 / 10
RATIO_TRAIN = 0.70
RATIO_VALID = 0.20
RATIO_TEST  = 0.10

SEED = 42   # reproductibilité

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# Helpers

def log(msg): print(f"  {msg}")

def find_image_for_label(label_path: Path, images_dir: Path):
    """Retrouve l'image correspondant à un fichier label .txt."""
    stem = label_path.stem
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def classes_in_label(label_path: Path):
    """Retourne l'ensemble des identifiants de classes présents dans un label."""
    found = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            found.add(int(line.split()[0]))
    return found


# 1. Collecte de toutes les images des 3 splits d'origine

def gather_samples():
    print("\n── Collecte des images d'origine ──")
    samples = []   # liste de (label_path, image_path, vecteur_classes)

    for split in ("train", "valid", "test"):
        lbl_dir = SRC_DIR / split / "labels"
        img_dir = SRC_DIR / split / "images"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            img = find_image_for_label(lbl, img_dir)
            if img is None:
                continue   # label sans image → ignoré
            present = classes_in_label(lbl)
            if not present:
                continue   # image sans annotation → ignorée
            vec = np.zeros(NC, dtype=int)
            for c in present:
                vec[c] = 1
            samples.append((lbl, img, vec))

    log(f"Total images annotées collectées : {len(samples)}")
    return samples


# 2. Stratification multi-label en 3 ensembles

def stratified_split(samples):
    print("\n── Stratification multi-label (70/20/10) ──")
    X = np.arange(len(samples)).reshape(-1, 1)
    Y = np.array([s[2] for s in samples])

    # Étape A : on isole d'abord le TRAIN (70%) du reste (30%)
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(RATIO_VALID + RATIO_TEST), random_state=SEED
    )
    train_idx, rest_idx = next(msss1.split(X, Y))

    # Étape B : on coupe le reste (30%) en valid/test → 20/10, soit 2/3 vs 1/3
    rest_X = X[rest_idx]
    rest_Y = Y[rest_idx]
    test_share = RATIO_TEST / (RATIO_VALID + RATIO_TEST)   # 0.10 / 0.30 = 1/3
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_share, random_state=SEED
    )
    valid_local, test_local = next(msss2.split(rest_X, rest_Y))

    valid_idx = rest_idx[valid_local]
    test_idx  = rest_idx[test_local]

    splits = {
        "train": [samples[i] for i in train_idx],
        "valid": [samples[i] for i in valid_idx],
        "test":  [samples[i] for i in test_idx],
    }
    for name, items in splits.items():
        log(f"{name:6s} → {len(items)} images")
    return splits


# 3. Copie des fichiers vers le nouveau dataset

def copy_split(splits):
    print("\n Copie des fichiers")
    if DST_DIR.exists():
        raise SystemExit(
            f"Le dossier {DST_DIR} existe déjà. Supprime-le ou renomme-le avant de relancer."
        )

    for split, items in splits.items():
        (DST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        for lbl, img, _ in items:
            shutil.copy2(img, DST_DIR / split / "images" / img.name)
            shutil.copy2(lbl, DST_DIR / split / "labels" / lbl.name)
        log(f"{split:6s} copié")


# 4. data.yaml + vérification de la nouvelle répartition

def write_yaml_and_verify(splits):
    print("\n── data.yaml + vérification ──")

    # data.yaml du nouveau dataset
    yaml_content = (
        f"train: {(DST_DIR / 'train' / 'images').resolve()}\n"
        f"val: {(DST_DIR / 'valid' / 'images').resolve()}\n"
        f"test: {(DST_DIR / 'test' / 'images').resolve()}\n"
        f"nc: {NC}\n"
        f"names: {CLASSES}\n"
    )
    (DST_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")
    log(f"data.yaml écrit → {DST_DIR / 'data.yaml'}")

    # Comptage des instances par classe et par split
    print("\n  Répartition des instances par classe :")
    print(f"  {'Classe':10s} | {'train':>7s} | {'valid':>7s} | {'test':>7s}")
    print("  " + "-" * 40)
    per = {s: Counter() for s in ("train", "valid", "test")}
    for split, items in splits.items():
        for lbl, _, _ in items:
            for line in lbl.read_text().splitlines():
                if line.strip():
                    per[split][int(line.split()[0])] += 1
    for i, name in enumerate(CLASSES):
        print(f"  {name:10s} | {per['train'][i]:7d} | {per['valid'][i]:7d} | {per['test'][i]:7d}")


# Main

def main():
    print("█" * 55)
    print("  RE-SPLIT STRATIFIÉ : Détection d'armes")
    print("█" * 55)

    if not SRC_DIR.exists():
        raise SystemExit(f"Dataset source introuvable : {SRC_DIR}")

    samples = gather_samples()
    if not samples:
        raise SystemExit("Aucune image annotée trouvée, vérifie le dossier source.")

    splits = stratified_split(samples)
    copy_split(splits)
    write_yaml_and_verify(splits)

    print("\n  ✓ Terminé. Nouveau dataset prêt dans :", DST_DIR.resolve())
    print("    → Pour l'utiliser : pointe DATASET_DIR sur ce dossier dans training.py,")
    print("      puis réentraîne (SANS --skip-train).\n")


if __name__ == "__main__":
    main()