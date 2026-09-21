"""
Interface web de détection d'armes YOLOv8 + Gradio
Mémoire : L'IA dans l'investigation criminelle

Trois modes dans la même interface :
    • Image  : on dépose une photo → image annotée + liste des détections
    • Vidéo  : on dépose une vidéo → vidéo annotée téléchargeable
    • Webcam : flux en direct annoté en temps réel

Prérequis :
    pip install gradio ultralytics opencv-python

Lancement :
    python app.py
    puis ouvre l'adresse affichée (http://127.0.0.1:7860) dans ton navigateur.
"""

from pathlib import Path
import csv
import time
import cv2
import gradio as gr
from ultralytics import YOLO


# Configuration

WEIGHTS = "runs/detect/results/weapon_detector_v1-2/weights/best.pt"  # ← ton modèle
CLASSES = ['Grenade', 'Knife', 'Missile', 'Pistol', 'Rifle']

# Couleurs par classe (format RGB pour l'affichage)
CLASS_COLORS = {
    "Grenade": (220,  40,  40),
    "Knife":   (255, 140,   0),
    "Missile": (180,  60, 200),
    "Pistol":  (40,  120, 240),
    "Rifle":   (40,  180, 100),
}
DEFAULT_COLOR = (200, 0, 0)


# Chargement du modèle (une seule fois au démarrage)

if not Path(WEIGHTS).exists():
    raise SystemExit(
        f"Modèle introuvable : {WEIGHTS}\n"
        "Vérifie le chemin de ton best.pt dans la variable WEIGHTS."
    )

print("Chargement du modèle…")
model = YOLO(WEIGHTS)
print("Modèle prêt.")


# Fonction d'annotation commune

def annotate(frame_bgr, conf_threshold):
    """Dessine les boîtes + labels sur une frame (BGR OpenCV). Renvoie (frame, détections)."""
    results = model(frame_bgr, conf=conf_threshold, verbose=False)
    detections = []

    for r in results:
        for box in (r.boxes or []):
            conf = float(box.conf[0])
            cls_name = r.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Couleur BGR (OpenCV) à partir du RGB défini plus haut
            r_, g_, b_ = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            color = (b_, g_, r_)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - th - bl - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1 + 3, y1 - bl - 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append((cls_name, conf))

    return frame_bgr, detections


def detections_summary(detections):
    """Texte récapitulatif des détections, trié par confiance décroissante."""
    if not detections:
        return "Aucune arme détectée."
    lines = ["**Détections :**"]
    for cls_name, conf in sorted(detections, key=lambda d: -d[1]):
        lines.append(f"- {cls_name} — {conf:.0%}")
    return "\n".join(lines)


# Mode IMAGE

def process_image(image_rgb, conf_threshold):
    if image_rgb is None:
        return None, "Dépose une image pour lancer la détection."
    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    annotated, dets = annotate(frame_bgr, conf_threshold)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections_summary(dets)


# Mode VIDÉO

def _fmt_time(seconds):
    """Convertit un nombre de secondes en HH:MM:SS."""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def process_video(video_path, conf_threshold, frame_step, progress=gr.Progress()):
    if video_path is None:
        return None, "Dépose une vidéo pour lancer la détection.", None

    frame_step = max(1, int(frame_step))   # sécurité : au moins 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Impossible d'ouvrir la vidéo.", None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = str(Path(video_path).with_name("detection_output.mp4"))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    # Timeline : on regroupe les apparitions continues d'une même classe en un
    # seul "événement" (début, fin, confiance max) plutôt qu'une ligne par frame.
    # GAP_TOLERANCE est exprimé en secondes et tient compte de l'échantillonnage.
    GAP_TOLERANCE = max(1.0, 2 * frame_step / fps)
    active = {}                  # classe → {start, last_seen, conf_max} en cours
    events = []                  # événements terminés : (classe, t_début, conf_max)

    def close_event(cls_name, info):
        events.append((cls_name, info["start"], info["conf_max"]))

    frame_id = 0
    last_annotated = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_id / fps        # temps de cette frame, en secondes

        # On n'analyse qu'une frame sur "frame_step". Les frames intermédiaires
        # réutilisent la dernière annotation pour garder une vidéo fluide.
        if frame_id % frame_step == 0:
            annotated, dets = annotate(frame, conf_threshold)
            last_annotated = annotated

            seen_now = {}
            for cls_name, conf in dets:
                seen_now[cls_name] = max(seen_now.get(cls_name, 0), conf)

            for cls_name, conf in seen_now.items():
                if cls_name in active:
                    active[cls_name]["last_seen"] = t
                    active[cls_name]["conf_max"] = max(active[cls_name]["conf_max"], conf)
                else:
                    active[cls_name] = {"start": t, "last_seen": t, "conf_max": conf}

            for cls_name in list(active.keys()):
                if cls_name not in seen_now and (t - active[cls_name]["last_seen"]) > GAP_TOLERANCE:
                    close_event(cls_name, active.pop(cls_name))

            writer.write(annotated)
        else:
            # Frame non analysée : on réécrit la dernière image annotée (ou l'originale)
            writer.write(last_annotated if last_annotated is not None else frame)

        frame_id += 1
        if total:
            progress(frame_id / total, desc=f"Frame {frame_id}/{total}")

    # Ferme les événements encore actifs en fin de vidéo
    for cls_name, info in active.items():
        close_event(cls_name, info)

    cap.release()
    writer.release()

    events.sort(key=lambda e: e[1])
    if not events:
        return out_path, "Aucune arme détectée dans la vidéo.", None

    # Tableau Markdown affiché à l'écran
    lines = ["### Chronologie des détections", "",
             "| Temps | Arme | Confiance max |", "|---|---|---|"]
    for cls_name, t_start, conf_max in events:
        lines.append(f"| {_fmt_time(t_start)} | {cls_name} | {conf_max:.0%} |")
    summary = "\n".join(lines)

    # Fichier CSV téléchargeable
    csv_path = str(Path(video_path).with_name("chronologie_detections.csv"))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f, delimiter=";")
        wcsv.writerow(["Temps", "Arme", "Confiance_max"])
        for cls_name, t_start, conf_max in events:
            wcsv.writerow([_fmt_time(t_start), cls_name, f"{conf_max:.2%}"])

    return out_path, summary, csv_path


# Mode WEBCAM (temps réel)

def process_webcam(frame_rgb, conf_threshold):
    if frame_rgb is None:
        return None
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated, _ = annotate(frame_bgr, conf_threshold)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# Construction de l'interface

with gr.Blocks(title="Détection d'armes IA", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Détection d'armes par IA\n"
        "Modèle YOLOv8 — classes : **Grenade, Knife, Missile, Pistol, Rifle**\n\n"
        "*Mémoire : L'IA dans l'investigation criminelle*"
    )

    with gr.Row():
        conf_slider = gr.Slider(
            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
            label="Seuil de confiance",
            info="Plus haut = moins de détections mais plus sûres"
        )

    with gr.Tabs():
        # Onglet Image
        with gr.Tab("Image"):
            with gr.Row():
                img_in = gr.Image(type="numpy", label="Image à analyser")
                img_out = gr.Image(type="numpy", label="Résultat")
            img_txt = gr.Markdown()
            gr.Button("Analyser", variant="primary").click(
                process_image, [img_in, conf_slider], [img_out, img_txt]
            )

        # Onglet Vidéo
        with gr.Tab("Vidéo"):
            step_slider = gr.Slider(
                minimum=1, maximum=60, value=5, step=1,
                label="Analyser 1 frame sur N",
                info="1 = toutes les frames (précis, lent). 5-10 conseillé. "
                     "Pour une vidéo longue (30 min+), monter à 15-30."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    vid_in = gr.Video(label="Vidéo à analyser")
                    vid_out = gr.Video(label="Vidéo annotée")
                with gr.Column(scale=1):
                    vid_txt = gr.Markdown("*La chronologie des détections s'affichera ici.*")
                    vid_csv = gr.File(label="Exporter la chronologie (CSV)")
            gr.Button("Analyser la vidéo", variant="primary").click(
                process_video, [vid_in, conf_slider, step_slider], [vid_out, vid_txt, vid_csv]
            )

        # Onglet Webcam
        with gr.Tab("Webcam (live)"):
            gr.Markdown("Autorise l'accès à ta caméra. La détection se fait en direct.")
            cam_in = gr.Image(sources=["webcam"], streaming=True, label="Caméra")
            cam_out = gr.Image(label="Détection en direct")
            cam_in.stream(
                process_webcam, [cam_in, conf_slider], [cam_out],
                stream_every=0.1,   # ~10 images/seconde
            )


if __name__ == "__main__":
    demo.launch()