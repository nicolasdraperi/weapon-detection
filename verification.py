from pathlib import Path
from collections import Counter, defaultdict

DATASETS = {
    "AVANT (origine)": "weapon-detection-1",
    "APRES (re-split)": "weapon-detection-resplit-origin",
}

SPLITS = ("train", "valid", "test")
DEFAULT_NAMES = ['Grenade', 'Knife', 'Missile', 'Pistol', 'Rifle']


def lire_noms(racine: Path):
    """Récupère l'ordre des classes depuis data.yaml, sinon liste par défaut."""
    yaml_path = racine / "data.yaml"
    if not yaml_path.exists():
        return DEFAULT_NAMES
    txt = yaml_path.read_text(encoding="utf-8", errors="ignore")
    # parse minimal : names: ['A', 'B', ...]  ou  names:\n  - A\n  - B
    import re
    m = re.search(r"names\s*:\s*\[(.*?)\]", txt, re.S)
    if m:
        return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    noms = re.findall(r"^\s*-\s*(.+?)\s*$", txt, re.M)
    return [n.strip().strip("'\"") for n in noms] if noms else DEFAULT_NAMES


def compter(racine: Path, noms):
    """Retourne instances[split][classe] et images[split][classe]."""
    instances = defaultdict(Counter)   # nb de boîtes
    images = defaultdict(Counter)      # nb d'images contenant la classe
    nb_images_split = Counter()

    for split in SPLITS:
        dossier = racine / split / "labels"
        if not dossier.exists():
            print(f"  [!] Dossier introuvable : {dossier}")
            continue
        for f in dossier.glob("*.txt"):
            nb_images_split[split] += 1
            classes_dans_image = set()
            for line in f.read_text().splitlines():
                if line.strip():
                    cid = int(line.split()[0])
                    if 0 <= cid < len(noms):
                        instances[split][noms[cid]] += 1
                        classes_dans_image.add(noms[cid])
            for c in classes_dans_image:
                images[split][c] += 1
    return instances, images, nb_images_split


def afficher(label, racine_str):
    racine = Path(racine_str)
    print("=" * 70)
    print(f"  {label}  →  {racine}")
    print("=" * 70)
    if not racine.exists():
        print(f"  [!] Dataset introuvable, ignoré.\n")
        return
    noms = lire_noms(racine)
    instances, images, nb_img = compter(racine, noms)

    # Tableau INSTANCES (boîtes)
    print(f"\n  INSTANCES (boîtes) par classe et par split")
    entete = f"  {'Classe':10s} | " + " | ".join(f"{s:>7s}" for s in SPLITS) + " |   Total"
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    for c in noms:
        vals = [instances[s][c] for s in SPLITS]
        print(f"  {c:10s} | " + " | ".join(f"{v:7d}" for v in vals) + f" | {sum(vals):7d}")
    tot = [sum(instances[s][c] for c in noms) for s in SPLITS]
    print(f"  {'TOTAL':10s} | " + " | ".join(f"{v:7d}" for v in tot) + f" | {sum(tot):7d}")

    # Tableau IMAGES
    print(f"\n  IMAGES contenant la classe, par split")
    for c in noms:
        vals = [images[s][c] for s in SPLITS]
        print(f"  {c:10s} | " + " | ".join(f"{v:7d}" for v in vals) + f" | {sum(vals):7d}")
    print(f"  {'Nb images':10s} | " + " | ".join(f"{nb_img[s]:7d}" for s in SPLITS)
          + f" | {sum(nb_img.values()):7d}")
    print()


if __name__ == "__main__":
    for label, chemin in DATASETS.items():
        afficher(label, chemin)