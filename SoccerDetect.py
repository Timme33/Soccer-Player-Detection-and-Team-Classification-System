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
    """
    Overlay semi-transparent line segments on top of an image using PIL.

    Why PIL here (instead of OpenCV):
      - PIL makes it easy to draw with alpha transparency (RGBA overlay) and then
        alpha-composite the overlay onto the base image.
      - The overlay approach (draw on transparent layer, composite once) avoids
        repeatedly blending into the base image for each segment.

    Parameters
    ----------
    image : str | PIL.Image.Image
        Either a filesystem path to an image or an already-loaded PIL Image.
        In the backend pipeline we often pass in a path to an intermediate file.
    segments : Iterable[((x1, y1), (x2, y2))]
        Iterable of line segments. Each segment connects two (x, y) points in image coordinates.
    line_width : int
        Thickness of the rendered lines in pixels.
    alpha : int
        Line transparency. 0 = fully transparent, 255 = fully opaque.
        Using semi-transparency helps lines remain visible without obscuring the frame.

    Returns
    -------
    PIL.Image.Image
        The annotated image. Returned as RGB for convenience unless the original had alpha.
    """
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
    """
    Compute the *circular* distance between two OpenCV hue values.

    OpenCV stores Hue in the range [0, 179] (not [0, 360]) where hue wraps around:
      - hue=0 and hue=179 are adjacent colors.
    A normal absolute difference would incorrectly treat values near the wrap-around
    as far apart (e.g., 1 vs 179). This function fixes that by taking the shorter
    distance around the hue circle.

    Parameters
    ----------
    h1, h2 : int | float
        Hue values in OpenCV's scale [0, 179].

    Returns
    -------
    float
        Minimum wrap-around hue distance in [0, 90] (since the maximum shortest distance is half-circle).
    """
    d = np.abs(h1 - h2)
    return np.minimum(d, 180 - d)

def estimate_grass_hue(image_bgr, boxes, sample_frac=0.1):
    """
    Estimate the dominant grass hue from the image background (outside detected boxes).

    Motivation:
      - Team colors are often extracted from player crops.
      - The biggest failure mode is accidentally picking grass pixels as the "dominant"
        color, especially near ankles/shorts or when segmentation leaks into the field.
      - We compute a robust estimate of the field (grass) hue so downstream logic can
        bias away from "grass-like" colors.

    Strategy:
      1) Create a boolean mask for "background" pixels by excluding all detection boxes.
      2) Randomly subsample background pixels for speed (large images can have millions of pixels).
      3) Convert sampled pixels to HSV.
      4) Prefer *more saturated* pixels (S > 40) because they are more likely to represent
         the grass color than white lines, shadows, or low-saturation areas.
      5) Build a hue histogram and return the most frequent hue bin (mode).

    Notes:
      - This is intentionally lightweight and unsupervised.
      - Works well across different cameras/lighting because it uses a mode, not a mean.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in OpenCV BGR format.
    boxes : list[(x1, y1, x2, y2)]
        Bounding boxes to exclude when sampling background pixels.
    sample_frac : float
        Fraction of eligible background pixels to sample (bounded by a minimum sample size).

    Returns
    -------
    int
        Estimated grass hue in OpenCV hue units [0, 179].
    """
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
    """
    Compute a coarse foreground mask for a cropped detection using GrabCut.

    Motivation:
      - YOLO boxes contain both player pixels and background pixels (grass, lines, ads).
      - For team classification we want *jersey pixels*, not the entire crop.
      - GrabCut provides a quick, classical segmentation method that separates likely
        foreground (player) from background without training a new model.

    How it works:
      - Initialize GrabCut with a rectangle slightly inset from the crop boundary.
        Pixels outside the rectangle are treated as probable background.
      - Run a few GrabCut iterations to refine foreground/background labeling.
      - Convert GrabCut's multi-class mask into a binary mask:
          1 = likely foreground, 0 = likely background.
      - Apply light smoothing and dilation to reduce speckle noise and recover thin jersey regions.

    Failure handling:
      - If GrabCut fails (rare but possible on tiny crops or degenerate inputs),
        return an all-ones mask so the pipeline can continue (fallback behavior).

    Parameters
    ----------
    crop : np.ndarray
        Cropped BGR image (player bounding box region).

    Returns
    -------
    np.ndarray
        Binary mask (uint8) with shape (H, W), values in {0, 1}.
    """
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
    """
    From a set of HSV pixels (ideally from the player's foreground mask), select the subset
    that most likely corresponds to the jersey color.

    Key problem:
      - Even within a foreground mask we can still have non-jersey pixels:
          shorts, socks, skin, shadows, bits of grass/lines leaking through segmentation.
      - Jerseys are often more saturated and less "grass-like" than the field.

    Approach (unsupervised, per-player):
      1) If we have too few pixels, return a robust median HSV of whatever we have.
      2) Otherwise, embed hue as (cos(theta), sin(theta)) to handle hue wrap-around,
         and include normalized S and V as additional cues.
      3) Run KMeans with k=2 *within this player's pixel set*.
      4) Score each cluster by:
           - higher median saturation (likely fabric color rather than skin/gray shadows),
           - larger circular distance from the estimated grass hue (avoid grass leakage).
      5) Return the median HSV of the higher-scoring cluster as the player's jersey color.

    Why the cosine/sine embedding:
      - Hue is circular; using trig embedding prevents discontinuities at hue wrap-around.

    Parameters
    ----------
    hsv_pixels : np.ndarray
        Array of HSV pixels with shape (N, 3).
    grass_hue : int
        Estimated field hue (OpenCV units [0, 179]) to penalize grass-like clusters.

    Returns
    -------
    np.ndarray
        Median HSV vector (H, S, V) representing the chosen jersey-color cluster.
    """
    if hsv_pixels.shape[0] < 50:
        return np.median(hsv_pixels, axis=0)
    H, S, V = hsv_pixels[:,0], hsv_pixels[:,1], hsv_pixels[:,2]
    theta = (H / 180.0) * 2.0 * np.pi
    feats = np.stack([np.cos(theta), np.sin(theta), S/255.0, V/255.0], axis=1)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats) #KMeans clustering Algorithm
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
    """
    End-to-end computer vision pipeline for a single user-uploaded soccer image.

    This function is designed to be called from the backend after the frontend uploads an image
    (typically saved to disk temporarily). It performs player detection, unsupervised team
    classification based on kit colors, and produces two visualization images (one per team)
    with bounding boxes and simple connecting lines.

    High-level flow
    ---------------
    1) Load the soccer-specific YOLOv8 detector and run inference on the full frame.
       Output: a list of detections (boxes, class ids, confidences).

    2) Estimate the grass hue from pixels outside detection boxes.
       Purpose: reduce the chance that grass is selected as a player's “dominant” color.

    3) For each player-like detection:
         a) Crop the bounding box region from the frame.
         b) Run GrabCut to obtain a coarse foreground mask (player vs background).
         c) Apply simple band-removal heuristics (remove top/bottom parts) to emphasize jersey pixels.
         d) Convert the crop to HSV and gather candidate pixels from the masked region.
         e) Run a small k=2 clustering step within that player's pixels and choose the cluster that is
            more likely to represent the jersey color (higher saturation, farther from grass hue).
       Output: one representative HSV color vector per detected player.

    4) Cluster all player jersey-color feature vectors into 3 groups using KMeans:
         - Team A
         - Team B
         - Other (referee/goalkeeper/noise)
       The two largest clusters are assumed to be the teams; the smallest is treated as “Other”.

    5) Visualization (two separate images):
         - Create a Team A image and a Team B image (copies of the original frame).
         - Draw gray bounding boxes only for that team (ignore “Other”).
         - Compute a stable anchor point per player (bottom-center of the bounding box).
         - Build a simple set of connecting segments by sorting anchors left-to-right.
         - Use `annotate_segments` (PIL overlay + alpha composite) to draw semi-transparent lines.
       Output: two annotated image files written to disk and returned as paths.

    Important notes / assumptions
    -----------------------------
    - Unsupervised classification: This does NOT assume fixed kit colors (works for arbitrary colors).
      Team labels are relative (“Team A” vs “Team B”) and may swap across images.
    - The “Other” cluster is heuristic: it typically captures referees/goalkeepers/odd detections, but
      depends on the image content and detection quality.
    - This is single-frame only (no tracking). Connecting lines are a visualization convenience, not a
      tactical graph; for video you would add temporal tracking/persistence.
    - Performance: For a real backend, you would usually load the YOLO model once at process startup
      and reuse it across requests; loading inside this function is simpler but slower.
    - File outputs: This implementation writes debug artifacts and output images to local disk.
      In production, you may want per-request unique paths, a temp directory, or object storage.

    Parameters
    ----------
    image_path : str
        Filesystem path to the uploaded image (already saved by the backend).

    Returns
    -------
    tuple[str | None, str | None]
        (teamA_output_path, teamB_output_path) on success.
        (None, None) if insufficient player samples are available for clustering or if a failure is
        detected that prevents producing meaningful outputs.
    """

    print("Loading soccer-trained YOLO model...")
    detector = YOLO("player_detection_best.pt") # Load the soccer-trained detection model (in production this would typically be cached globally).

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    print("Running detection...")
    res = detector(frame, conf=0.5)[0] # Run object detection on the uploaded frame and collect all predicted bounding boxes.

    all_boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0]); all_boxes.append((x1, y1, x2, y2))
    grass_h = estimate_grass_hue(frame, all_boxes) # Estimate dominant field (grass) hue so jersey color extraction can avoid grass contamination.
    print(f"Estimated grass hue ≈ {grass_h}")

    player_boxes, player_colors = [], []
    os.makedirs("mask_debug_full", exist_ok=True)

    names = detector.names
    # For each detected player, isolate foreground pixels and extract a representative jersey color.
    for idx, b in enumerate(res.boxes):
        cls = int(b.cls[0])
        label = names[cls].lower() if isinstance(names, dict) else str(names[cls]).lower()
        if "player" not in label and "person" not in label:
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        mask = grabcut_mask(crop) # rough foreground mask to isolate the player from background
        # Remove top/bottom bands: reduces bias from faces/shorts/field and emphasizes jersey region.
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
        # Return placeholders or indicate failure
        return None, None

    player_colors = np.array(player_colors, dtype=np.float32)
    H, S, V = player_colors[:,0], player_colors[:,1], player_colors[:,2]
    theta = (H / 180.0) * 2.0 * np.pi
    hx, hy = np.cos(theta), np.sin(theta)
    features = np.stack([hx, hy, (S/255.0)*0.45, (V/255.0)*0.25], axis=1)

    # KMeans with 3 clusters: two teams + a catch-all "other" cluster (refs, goalies, noise, etc.)
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
        cx, cy = (x1 + x2)//2, y2  # bottom-center anchor for connecting lines (more stable than box center)
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
