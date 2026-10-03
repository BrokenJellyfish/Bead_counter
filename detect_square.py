"""
Locate the large counting square (~fixed size) by fitting its 4 edges to the
chamber grid lines. Size is known from the labelled examples; only position and
a small rotation vary between photos.

Method:
  1. Build horizontal/vertical grid-line strength maps (blackhat -> long-run
     morphological opening; keeps grid lines, drops the compact beads).
  2. For each candidate rotation theta, rotate the line maps and take row/column
     projections. A square of side S placed at (x0,y0) scores by how much its
     top/bottom rows and left/right columns overlap detected lines.
  3. The score is separable per axis, so search is cheap. Keep the best
     (theta, x0, y0, S). Edges are re-scored within the square's own extent to
     avoid latching onto lines that are only strong elsewhere.
"""

import numpy as np
import cv2

S_DEFAULT = 1160          # square side in px (mean of the two labelled examples)
S_RANGE = range(1120, 1205, 15)
ANGLE_RANGE = np.arange(-15.0, 15.01, 0.5)   # grid can be hand-placed at any tilt
BLACKHAT_K = 31
LINE_THRESH = 12
OPEN_LEN = 60


BOUNDARY_W = 2.5          # weight of triple-boundary evidence vs plain grid lines


def line_maps(gray):
    """Return (hor, ver) plain grid-line masks and (bhor, bver) thick triple-
    boundary masks. Plain masks keep any long line; boundary masks keep only
    bands where several parallel lines cluster (the large-square borders)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_K, BLACKHAT_K))
    bh = cv2.subtract(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k), gray)
    _, m = cv2.threshold(bh, LINE_THRESH, 255, cv2.THRESH_BINARY)
    hor = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_LEN, 1)))
    ver = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, OPEN_LEN)))
    # thick boundaries: close perpendicular to merge nearby parallel lines,
    # then open to erase isolated single lines.
    bhor = cv2.morphologyEx(hor, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    bhor = cv2.morphologyEx(bhor, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)))
    bver = cv2.morphologyEx(ver, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    bver = cv2.morphologyEx(bver, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    f = lambda a: (a > 0).astype(np.float32)
    return f(hor), f(ver), f(bhor), f(bver)


def _rotate(img, theta, center):
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_NEAREST), M


def _norm(a):
    m = a.max()
    return a / m if m > 0 else a


def grid_angle(hor, ver, center, H, W):
    """Grid tilt, found by projection sharpness: when the line map is de-rotated
    to the true angle, all lines land in the same rows/cols, so the row/column
    projections have sharp regular peaks (high variance). Robust even in the
    dense centre square where the edge-overlap fit alone can lock onto a wrong
    angle."""
    best = None
    for theta in ANGLE_RANGE:
        horr, _ = _rotate(hor, theta, center)
        verr, _ = _rotate(ver, theta, center)
        sharp = horr.sum(1).var() + verr.sum(0).var()
        if best is None or sharp > best[1]:
            best = (theta, sharp)
    return best[0]


def best_square(gray):
    H, W = gray.shape
    hor, ver, bhor, bver = line_maps(gray)
    center = (W / 2, H / 2)
    # Stage 1: fix the rotation robustly by sharpness. Stage 2: fit size and
    # position over a small window around it (edge-overlap can nudge +/-1 deg).
    theta0 = grid_angle(hor, ver, center, H, W)
    best = None
    for theta in np.arange(theta0 - 1.0, theta0 + 1.01, 0.5):
        horr, M = _rotate(hor, theta, center)
        verr, _ = _rotate(ver, theta, center)
        bhorr, _ = _rotate(bhor, theta, center)
        bverr, _ = _rotate(bver, theta, center)
        # combine plain-line and boundary evidence (each normalised, boundary
        # weighted up so a triple line anchors an edge but a single line can
        # still complete the square via the fixed size).
        rowproj = _norm(horr.sum(1)) + BOUNDARY_W * _norm(bhorr.sum(1))
        colproj = _norm(verr.sum(0)) + BOUNDARY_W * _norm(bverr.sum(0))
        for S in S_RANGE:
            if S >= W or S >= H:
                continue
            ys = np.arange(0, H - S)
            xs = np.arange(0, W - S)
            y0 = ys[int(np.argmax(rowproj[ys] + rowproj[ys + S]))]
            x0 = xs[int(np.argmax(colproj[xs] + colproj[xs + S]))]
            score = (rowproj[y0] + rowproj[y0 + S]
                     + colproj[x0] + colproj[x0 + S])
            if best is None or score > best[0]:
                best = (score, theta, x0, y0, S, M)
    score, theta, x0, y0, S, M = best
    # corners in the rotated frame -> map back to original with inverse rotation
    corners_rot = np.array([[x0, y0], [x0 + S, y0],
                            [x0 + S, y0 + S], [x0, y0 + S]], dtype=np.float32)
    Minv = cv2.invertAffineTransform(M)
    corners = cv2.transform(corners_rot[None], Minv)[0]
    return dict(score=float(score), theta=float(theta), side=int(S),
                corners=corners, axis_box=(int(x0), int(y0), int(S)))


def verify_subdivision(gray, result, n_cells, tol=12, min_frac=0.6):
    """Cheap self-check: does the square contain an n_cells x n_cells grid?

    Samples the grid-line projections at the expected internal divisions
    (i/n of the side, i=1..n-1) and checks a line is present there. Returns
    (ok, frac_found). n_cells = 4 for corner squares, 5 for the centre square.
    """
    H, W = gray.shape
    x0, y0, S = result["axis_box"]
    hor, ver, bhor, bver = line_maps(gray)
    horr, _ = _rotate(hor, result["theta"], (W / 2, H / 2))
    verr, _ = _rotate(ver, result["theta"], (W / 2, H / 2))
    colproj = verr[y0:y0 + S, :].sum(axis=0)   # per col x (len W), rows in square
    rowproj = horr[:, x0:x0 + S].sum(axis=1)    # per row y (len H), cols in square
    # border coverage sets the scale we compare internal lines against
    ref = np.median([colproj[x0], colproj[x0 + S], rowproj[y0], rowproj[y0 + S]])
    if ref <= 0:
        return False, 0.0
    found = 0
    total = 2 * (n_cells - 1)
    for i in range(1, n_cells):
        xi = x0 + round(i * S / n_cells)
        yi = y0 + round(i * S / n_cells)
        if colproj[max(0, xi - tol):xi + tol].max() >= 0.4 * ref:
            found += 1
        if rowproj[max(0, yi - tol):yi + tol].max() >= 0.4 * ref:
            found += 1
    frac = found / total
    return frac >= min_frac, float(frac)


def draw(img, corners, color=(0, 0, 255), t=3):
    out = img.copy()
    cv2.polylines(out, [corners.astype(np.int32)], True, color, t)
    return out


if __name__ == "__main__":
    import sys, os
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = best_square(gray)
        out = os.path.splitext(p)[0] + "_square.png"
        cv2.imwrite(out, draw(img, r["corners"]))
        print(f"{os.path.basename(p)}: side={r['side']} theta={r['theta']:.1f} "
              f"score={r['score']:.0f} corners={r['corners'].astype(int).tolist()} "
              f"-> {os.path.basename(out)}")
