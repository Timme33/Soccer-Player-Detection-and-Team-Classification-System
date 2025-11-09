from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from typing import Iterable, Tuple, Union, List
from PIL import Image, ImageDraw

# ---------- teammate's line function ----------
Point = Tuple[int, int]
Segment = Tuple[Point, Point]
ImageLike = Union[str, Image.Image]

def annotate_segments(
    image: ImageLike,
    segments: Iterable[Segment],
    line_width: int = 4,
    alpha: int = 128,
) -> Image.Image:
    base = Image.open(image) if isinstance(image, str) else image
    base_rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    color = (255, 255, 255, max(0, min(255, alpha)))
    for (x1, y1), (x2, y2) in list(segments):
        draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width, joint="curve")
    annotated = Image.alpha_composite(base_rgba, overlay)
    return annotated.convert("RGB") if base.mode in ("L", "RGB") else annotated


# ---------- helpers ----------
def circular_hue_dist(h1, h2):
    d = np.abs(h1 - h2)
    return np.minimum(d, 180 - d)

def estimate_grass_hue(image_bgr, boxes, sample_frac=0.1):
    mask = np.ones(image_bgr.shape[:2], dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        mask[y1:y2, x1:x2] = False
    ys, xs = np.where(mask)
    if len(ys) == 0:
        ys, xs = np.indices(mask.shape)
        ys, xs = ys.ravel(), xs.ravel()
    idx = np.random.choice(len(ys), size=max(2000, int(len(ys)*sample_frac)), replace=False)
    sample = image_bgr[ys[idx], xs[idx]]
    hsv = cv2.cvtColor(sample.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
    S, H = hsv[:,1], hsv[:,0]
    Hg = H[S > 40] if np.any(S > 40) else H
    hist, _ = np.histogram(Hg, bins=180, range=(0,180))
    return int(np.argmax(hist))

def grabcut_mask(crop):
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    rect = (2, 2, max(1, crop.shape[1]-4), max(1, crop.shape[0]-4))
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except:
        return np.ones(crop.shape[:2], np.uint8)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
    mask2 = cv2.medianBlur(mask2, 3)
    mask2 = cv2.dilate(mask2, np.ones((3,3), np.uint8), 1)
    return mask2

def pick_jersey_cluster(hsv_pixels, grass_hue):
    if hsv_pixels.shape[0] < 50:
        return np.median(hsv_pixels, axis=0)
    H, S, V = hsv_pixels[:,0], hsv_pixels[:,1], hsv_pixels[:,2]
    theta = (H / 180.0) * 2.0 * np.pi
    feats = np.stack([np.cos(theta), np.sin(theta), S/255.0, V/255.0], axis=1)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats)
    labels = km.labels_
    med0 = np.median(hsv_pixels[labels==0], axis=0)
    med1 = np.median(hsv_pixels[labels==1], axis=0)
    def score(med):
        h, s, v = med
        sat = s / 255.0
        hdist = circular_hue_dist(h, grass_hue)
        return 0.7*sat + 0.3*(hdist/90.0)
    return med0 if score(med0) >= score(med1) else med1


# ---------- main pipeline function ----------
def run_cv_pipeline(image_path: str):
    print("Loading soccer-trained YOLO model...")
    detector = YOLO("player_detection_best.pt")

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    print("Running detection...")
    res = detector(frame, conf=0.5)[0]

    all_boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0]); all_boxes.append((x1, y1, x2, y2))
    grass_h = estimate_grass_hue(frame, all_boxes)
    print(f"Estimated grass hue ≈ {grass_h}")

    player_boxes, player_colors = [], []
    os.makedirs("mask_debug_full", exist_ok=True)

    names = detector.names
    for idx, b in enumerate(res.boxes):
        cls = int(b.cls[0])
        label = names[cls].lower() if isinstance(names, dict) else str(names[cls]).lower()
        if "player" not in label and "person" not in label:
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        mask = grabcut_mask(crop)
        h = mask.shape[0]
        mask[:int(0.15*h), :] = 0
        mask[int(0.80*h):, :] = 0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        region = (mask > 0)
        px = hsv[region]
        if px.size == 0:
            px = hsv.reshape(-1,3)

        dom = pick_jersey_cluster(px, grass_h)
        if dom is not None:
            player_boxes.append((x1, y1, x2, y2))
            player_colors.append(dom)

            overlay = crop.copy()
            overlay[mask == 0] = (0,0,0)
            cv2.imwrite(f"mask_debug_full/player_{idx}.jpg", overlay)

    if len(player_colors) < 3:
        print("❌ Not enough player samples for clustering.")
        raise RuntimeError("Not enough player samples for clustering")

    player_colors = np.array(player_colors, dtype=np.float32)
    H, S, V = player_colors[:,0], player_colors[:,1], player_colors[:,2]
    theta = (H / 180.0) * 2.0 * np.pi
    hx, hy = np.cos(theta), np.sin(theta)
    features = np.stack([hx, hy, (S/255.0)*0.45, (V/255.0)*0.25], axis=1)

    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(features)
    labels = km.labels_
    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]
    teamA_id, teamB_id, other_id = order[0], order[1], order[2]
    final = np.where(labels == teamA_id, 0, np.where(labels == teamB_id, 1, 2))

    # ---------- visualization ----------
    gray = (180, 180, 180)
    teamA_img = frame.copy()
    teamB_img = frame.copy()

    teamA_centers, teamB_centers = [], []
    for (x1, y1, x2, y2), lab in zip(player_boxes, final):
        cx, cy = (x1 + x2)//2, y2  # bottom-center
        if lab == 0:
            cv2.rectangle(teamA_img, (x1, y1), (x2, y2), gray, 2)
            teamA_centers.append((cx, cy))
        elif lab == 1:
            cv2.rectangle(teamB_img, (x1, y1), (x2, y2), gray, 2)
            teamB_centers.append((cx, cy))
        # ignore "Other"

    teamA_centers.sort(key=lambda p: p[0])
    teamB_centers.sort(key=lambda p: p[0])

    teamA_segments = list(zip(teamA_centers, teamA_centers[1:])) if len(teamA_centers) > 1 else []
    teamB_segments = list(zip(teamB_centers, teamB_centers[1:])) if len(teamB_centers) > 1 else []

    cv2.imwrite("teamA_boxes.jpg", teamA_img)
    cv2.imwrite("teamB_boxes.jpg", teamB_img)

    annotate_segments("teamA_boxes.jpg", teamA_segments, line_width=4, alpha=140).save("teamA_boxes_lines.jpg")
    annotate_segments("teamB_boxes.jpg", teamB_segments, line_width=4, alpha=140).save("teamB_boxes_lines.jpg")

    print("✅ Saved → teamA_boxes_lines.jpg and teamB_boxes_lines.jpg (gray boxes + bottom-center connecting lines)")

    return "teamA_boxes_lines.jpg", "teamB_boxes_lines.jpg"


# ---------- local test ----------
if __name__ == "__main__":
    run_cv_pipeline("CityVSLiv.jpg")
