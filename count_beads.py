"""
Bead / black-particle counter for hemocytometer-style images.

Pipeline:
  1. Grayscale + threshold so dark beads are kept and faint grid lines drop out.
  2. Remove residual grid fragments by shape (thin / elongated / low-solidity).
  3. Connected-component analysis -> one blob per bead or clump.
  4. Auto-calibrate single-bead area per image = median blob area
     (after dropping noise specks and huge clumps).
  5. Aggregate rule:
        ratio = area / bead_area
        ratio <= SINGLE_MAX (1.5)            -> 1 bead
        SINGLE_MAX < ratio <= AGG_CAP (5)    -> round(ratio) beads
        ratio > AGG_CAP                      -> ignored (logged)
  6. Annotated image + per-image counts.
"""

import os
import glob
import csv
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima

# ---- detector selection -------------------------------------------------
# "dog"    = Fiji DoG + prominence Find-Maxima method (count_beads_masked_batch.ijm).
#            Each bead is one local maximum, so touching beads count separately
#            without any area-ratio guessing (fixes the couples/singles problem).
# "legacy" = the original black-hat + connected-component + area-ratio method.
DETECTOR = "dog"

# ---- DoG detector parameters (defaults from the spec / Fiji macro) ------
MASK_MAX   = 20     # green value <= this is mask (JPEG black is not exactly 0)
SIGMA_GATE = 1.0    # Gaussian sigma (px) for the intensity-gate image ("smooth")
SIGMA_1    = 1.4    # DoG sigma 1 (px), matched to ~5.5 px beads
SIGMA_2    = 2.8    # DoG sigma 2 (px)
PROMINENCE = 2.0    # min. prominence of a DoG maximum (green intensity units)
# Intensity gate: a maximum counts only if its smoothed green is clearly darker
# than the image background, i.e. gate_img[y,x] < GATE_FRAC * background. This is
# RELATIVE to each photo's own median background, not a fixed absolute value.
# Why: beads and faint grid lines form two separate intensity peaks (beads darker,
# ~0.4-0.6*bg; grid lines near ~0.7*bg). The old fixed GATE_MAX=72 sat in the
# MIDDLE of the bead peak, so it discarded ~half of the real beads. 0.65*bg sits
# in the valley between the two peaks: it keeps the beads and rejects the grid
# lines, and it adapts to photos with different lighting. (Was GATE_MAX=72.)
GATE_FRAC   = 0.65
# Shrink the counting square inward by ~ROI_ERODE_K/2 px before detecting, so the
# square's own dark border line is not picked up as a ring of fake beads once the
# gate is relaxed. The eroded band is the printed border line, not sample area.
ROI_ERODE_K = 16    # erosion kernel (px); 0 disables

# ---- tunable parameters (legacy detector) -------------------------------
BLACKHAT_K        = 25     # closing kernel (px); must exceed a bead but be < grid spacing
BLACKHAT_THRESH   = 45     # response above this = bead; faint grid lines fall below
MIN_BLOB_PX       = 8      # absolute floor: smaller blobs are sensor noise
NOISE_FRAC        = 0.40   # blobs smaller than NOISE_FRAC * bead_area are noise
SINGLE_MAX        = 1.8    # ratio <= this counts as a single bead (higher =
                           #   fewer oversized singles mislabelled as couples)
BEAD_AREA_SCALE   = 1.6    # multiply calibrated single-bead area; raise to make
                           #   clump counting more conservative (fewer over-counts).
                           #   ~1.6-1.8 suits sparse samples; ~1.2-1.4 for very
                           #   dense samples with many genuine clumps.
AGG_CAP           = 10     # ratio above this is ignored as un-countable
MIN_SOLIDITY      = 0.55   # below this a blob is treated as a grid fragment
MAX_ASPECT        = 6.0    # length/width above this = grid line fragment
# -------------------------------------------------------------------------


def threshold_beads(gray):
    """Return a binary mask (uint8 0/255) where beads are 255.

    Black-hat = closing(gray) - gray. Closing with a kernel larger than a bead
    fills the small dark beads, so the difference lights up exactly the small
    dark spots while the slowly-varying background and the thin grid lines
    (which the closing also bridges) stay near zero.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_K, BLACKHAT_K))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    blackhat = cv2.subtract(closed, gray)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    # Beads are much darker than the faint grey grid lines, so both show up in
    # the black-hat but the beads respond far more strongly. Thresholding high
    # enough keeps the beads (including any sitting ON a line, since they are
    # dark) while the grid lines fall away -- no line-subtraction needed, and
    # nothing bridges separate beads into false clumps.
    _, mask = cv2.threshold(blackhat, BLACKHAT_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def blob_stats(mask):
    """Connected components -> list of dicts with geometry."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):  # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bbox_area = max(w * h, 1)
        aspect = max(w, h) / max(min(w, h), 1)
        solidity = area / bbox_area  # cheap proxy (bbox fill fraction)
        blobs.append(dict(label=i, area=int(area), w=int(w), h=int(h),
                          x=int(x), y=int(y), cx=cents[i][0], cy=cents[i][1],
                          aspect=aspect, solidity=solidity))
    return blobs, labels


def is_grid_fragment(b, bead_area):
    """Heuristic: long thin low-fill blobs are leftover grid lines."""
    if b["aspect"] >= MAX_ASPECT and b["solidity"] < MIN_SOLIDITY:
        return True
    # very long relative to a bead but skinny
    if max(b["w"], b["h"]) > 4 * np.sqrt(bead_area) and b["solidity"] < 0.35:
        return True
    return False


def calibrate_bead_area(blobs):
    """Single-bead area = mode of the blob-area distribution.

    Most blobs are single beads, so the most common area is one bead. The mode
    is robust in dense images where touching beads inflate the median (there the
    median blob can be a pair, not a single). Computed from a histogram over the
    plausible single-bead range."""
    areas = np.array([b["area"] for b in blobs if b["area"] >= MIN_BLOB_PX])
    if areas.size == 0:
        return None
    hi = np.percentile(areas, 85)          # ignore the big-clump tail
    core = areas[areas <= max(hi, MIN_BLOB_PX + 1)]
    # histogram with a few-px bin; the peak bin centre is the single-bead size
    bins = np.arange(core.min(), core.max() + 4, 4)
    if bins.size < 2:
        return float(np.median(core))
    h, edges = np.histogram(core, bins=bins)
    peak = edges[h.argmax()] + 2           # bin centre
    return float(peak)


def _inside(region, cx, cy):
    """True if point is inside the region polygon (or if no region given)."""
    if region is None:
        return True
    return cv2.pointPolygonTest(region, (float(cx), float(cy)), False) >= 0


def apply_square_mask(img, region, fill=0):
    """Return a copy of img showing only the region polygon; everything outside
    is set to `fill` (black by default)."""
    region_np = np.asarray(region, dtype=np.int32)
    keep = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(keep, [region_np], 255)
    out = np.full_like(img, fill)
    out[keep > 0] = img[keep > 0]
    return out


# ==== Fiji DoG + prominence detector ====================================
# Port of count_beads_masked_batch.ijm. Interface matches count_image so it is
# a drop-in behind the DETECTOR switch; run_counts.py needs no changes.

def _refine_subpixel(dog, pts):
    """TrackMate-style sub-pixel centre from the 3x3 DoG neighbourhood.
    Offset delta = -H^-1 . grad, applied only if H is negative-definite and
    |delta| <= 0.5 px. Changes positions only, never the count.
    pts, return: (N,2) float arrays in (row, col)."""
    out = pts.copy()
    H, W = dog.shape
    for k in range(len(pts)):
        y, x = pts[k]
        yi, xi = int(round(y)), int(round(x))
        if yi < 1 or yi >= H - 1 or xi < 1 or xi >= W - 1:
            continue
        p = dog[yi - 1:yi + 2, xi - 1:xi + 2]
        Dx = (p[1, 2] - p[1, 0]) / 2.0
        Dy = (p[2, 1] - p[0, 1]) / 2.0
        Dxx = p[1, 2] - 2 * p[1, 1] + p[1, 0]
        Dyy = p[2, 1] - 2 * p[1, 1] + p[0, 1]
        Dxy = (p[2, 2] - p[2, 0] - p[0, 2] + p[0, 0]) / 4.0
        det = Dxx * Dyy - Dxy * Dxy
        if Dxx < 0 and det > 0:                      # negative-definite Hessian
            try:
                dxy = -np.linalg.solve(np.array([[Dxx, Dxy], [Dxy, Dyy]]),
                                       np.array([Dx, Dy]))   # [dcol, drow]
            except np.linalg.LinAlgError:
                continue
            if abs(dxy[0]) <= 0.5 and abs(dxy[1]) <= 0.5:
                out[k, 0] = y + dxy[1]               # row += drow
                out[k, 1] = x + dxy[0]               # col += dcol
    return out


def detect_beads_dog(green, keep):
    """Fiji DoG + prominence detector.

    green : 2-D float array, the green channel (0-255).
    keep  : bool mask of the ROI (the detected square); outside is blacked out
            so it drops below MASK_MAX, exactly as the macro's masked input.
    Returns (points_yx, raw_maxima):
      points_yx  = (N,2) float array of sub-pixel (row, col) after the gate,
      raw_maxima = number of maxima before the intensity gate (Fiji "Raw maxima").
    """
    if ROI_ERODE_K > 0:                              # drop the square's border line
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_ERODE_K, ROI_ERODE_K))
        keep = cv2.erode(keep.astype(np.uint8), k) > 0
    g = green.astype(np.float32).copy()
    g[~keep] = 0.0                                   # black outside the square
    M = g > MASK_MAX                                 # counting mask (== Fiji ROI)
    if not M.any():
        return np.empty((0, 2), float), 0

    bg = float(np.median(g[M]))
    g2 = g.copy()
    g2[~M] = bg                                      # neutralise masked area to median

    gate_img = ndi.gaussian_filter(g2, SIGMA_GATE, mode="nearest", truncate=4.0)

    inv = -g2                                         # beads are dark -> invert
    dog = (ndi.gaussian_filter(inv, SIGMA_1, mode="nearest", truncate=4.0)
           - ndi.gaussian_filter(inv, SIGMA_2, mode="nearest", truncate=4.0))

    # Prominence-based maxima (ImageJ Find Maxima). Scale->int64 so the
    # morphological reconstruction inside h_maxima is exact (no float epsilon).
    scale = 1000.0
    dog_i = np.round(dog * scale).astype(np.int64)
    hmax = h_maxima(dog_i, int(round(PROMINENCE * scale)),
                    footprint=np.ones((3, 3)))

    # one point per maximum region: the pixel nearest the region centroid
    # (ImageJ's plateau rule), 8-connectivity.
    lbl, nlab = ndi.label(hmax, structure=np.ones((3, 3)))
    pts = []
    if nlab:
        coms = ndi.center_of_mass(hmax, lbl, range(1, nlab + 1))
        for i, sl in enumerate(ndi.find_objects(lbl), start=1):
            ys, xs = np.nonzero(lbl[sl] == i)
            ys = ys + sl[0].start
            xs = xs + sl[1].start
            cy, cx = coms[i - 1]
            j = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            y, x = int(ys[j]), int(xs[j])
            if M[y, x]:                              # keep maxima inside the mask
                pts.append((y, x))

    pts = np.array(pts, float).reshape(-1, 2)
    raw_maxima = len(pts)

    # intensity gate: drop grid-line ridges (bright in the smoothed image).
    # Threshold is relative to this photo's background (see GATE_FRAC note above).
    if len(pts):
        yi = pts[:, 0].astype(int)
        xi = pts[:, 1].astype(int)
        pts = pts[gate_img[yi, xi] < GATE_FRAC * bg]

    pts = _refine_subpixel(dog, pts)
    return pts, raw_maxima


def _count_image_dog(path, region=None):
    """DoG detector wrapped in the same interface / return dict as count_image."""
    img = cv2.imread(path)
    green = img[:, :, 1].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    h, w = green.shape

    if region is not None:
        region_np = np.asarray(region, dtype=np.int32)
        keep = np.zeros((h, w), np.uint8)
        cv2.fillPoly(keep, [region_np], 255)
        keep = keep > 0
    else:
        region_np = None
        keep = np.ones((h, w), bool)

    pts, raw_maxima = detect_beads_dog(green, keep)
    count = len(pts)

    # annotate on the masked image (only the counting square visible), red circles
    if region_np is not None:
        annotated = apply_square_mask(img, region_np)
        cv2.polylines(annotated, [region_np], True, (0, 0, 255), 3)
    else:
        annotated = img.copy()
    for y, x in pts:
        cv2.circle(annotated, (int(round(x)), int(round(y))), 5, (0, 0, 255), 2)

    # Same schema as the legacy dict. DoG has no clumps: singles == count, the
    # aggregate fields are 0 and bead_area is blank (revisit reporting later).
    return dict(path=path, count=count, singles=count, agg_beads=0,
                agg_clumps=0, ignored=0, bead_area="", raw_maxima=raw_maxima,
                annotated=annotated)


def count_image(path, region=None):
    """Count beads. If region (Nx2 int corners) is given, only beads whose
    centroid falls inside that polygon are calibrated against and counted."""
    if DETECTOR == "dog":
        return _count_image_dog(path, region)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = threshold_beads(gray)
    blobs, _ = blob_stats(mask)

    region_np = None if region is None else np.asarray(region, dtype=np.int32)
    # calibrate bead size from beads inside the region (representative of it)
    in_blobs = [b for b in blobs if _inside(region_np, b["cx"], b["cy"])]
    bead_area = calibrate_bead_area(in_blobs)
    if bead_area is not None:
        bead_area *= BEAD_AREA_SCALE   # bias the single-bead size up so darker,
        #   larger single beads are not mistaken for clumps
    if bead_area is None:
        return dict(path=path, count=0, singles=0, agg_beads=0, agg_clumps=0,
                    ignored=0, bead_area=0, annotated=img)

    noise_floor = NOISE_FRAC * bead_area
    total = singles = agg_beads = agg_clumps = ignored = 0
    # annotate on the masked image (only the counting square is visible)
    if region_np is not None:
        annotated = apply_square_mask(img, region_np)
        cv2.polylines(annotated, [region_np], True, (0, 0, 255), 3)
    else:
        annotated = img.copy()

    for b in blobs:
        if b["area"] < MIN_BLOB_PX or b["area"] < noise_floor:
            continue  # noise speck
        if not _inside(region_np, b["cx"], b["cy"]):
            continue  # outside the counting square
        if is_grid_fragment(b, bead_area):
            continue  # grid line remnant
        ratio = b["area"] / bead_area
        cx, cy = int(round(b["cx"])), int(round(b["cy"]))
        r = int(max(4, round(np.sqrt(b["area"] / np.pi))))

        if ratio <= SINGLE_MAX:
            total += 1
            singles += 1
            cv2.circle(annotated, (cx, cy), r + 3, (0, 200, 0), 2)
        elif ratio <= AGG_CAP:
            c = int(round(ratio))
            total += c
            agg_beads += c
            agg_clumps += 1
            cv2.rectangle(annotated, (b["x"] - 2, b["y"] - 2),
                          (b["x"] + b["w"] + 2, b["y"] + b["h"] + 2),
                          (255, 140, 0), 2)
            cv2.putText(annotated, str(c), (b["x"], max(0, b["y"] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 140, 0), 2)
        else:
            ignored += 1
            cv2.rectangle(annotated, (b["x"] - 2, b["y"] - 2),
                          (b["x"] + b["w"] + 2, b["y"] + b["h"] + 2),
                          (0, 0, 255), 2)
            cv2.putText(annotated, "X", (b["x"], max(0, b["y"] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return dict(path=path, count=total, singles=singles, agg_beads=agg_beads,
                agg_clumps=agg_clumps, ignored=ignored,
                bead_area=round(bead_area, 1), annotated=annotated)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = count_image(p)
        out = os.path.splitext(p)[0] + "_annotated.jpg"
        cv2.imwrite(out, r["annotated"])
        print(f"{os.path.basename(p)}: count={r['count']} "
              f"(singles={r['singles']}, agg_beads={r.get('agg_beads',0)} "
              f"in {r.get('agg_clumps',0)} clumps, ignored={r['ignored']}), "
              f"bead_area={r['bead_area']}px -> {os.path.basename(out)}")
