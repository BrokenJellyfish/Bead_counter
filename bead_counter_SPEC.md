# Bead Counting — Technical Specification

## 1. Purpose & scope

Automated counting of black particles (beads) in microscope photos of a
haemocytometer counting chamber, to measure **bead concentration** per sample and
**recovery %** between process steps (e.g. bulk vs transferred fraction).

The system takes raw microscope photos organised by experimental condition and
produces, per condition:

- bead count per 1 mm² counting square (per photo),
- averages across repeats,
- concentration (beads/mL) and total beads in a user-defined sample volume,
- recovery % relative to a reference condition,
- annotated / masked demo images for visual verification.

**In scope:** detection of the counting square, restriction of counting to that
square, bead detection (Difference-of-Gaussians peak finding, or the legacy
area-based method), concentration and recovery maths, and reviewable image
outputs.

**Out of scope:** absolute sizing in microns, bead classification by type,
tracking across images, and any wet-lab dilution steps upstream of imaging.

## 2. Imaging & chamber assumptions

- Chamber: **Neubauer-improved** haemocytometer — a 3×3 grid of **1 mm × 1 mm
  large squares**. At the standard 0.1 mm chamber depth, one large square encloses
  **1 mm² × 0.1 mm = 1×10⁻⁴ mL**.
- The four **corner** large squares are subdivided into a **4×4** array of 0.25 mm
  medium cells; the **centre** large square is subdivided into a **5×5** array of
  0.20 mm cells (each further split into 16). This distinguishes corner vs centre
  squares in the images.
- Imaging convention: **5 photos per repeat** = the 4 corner squares followed by
  the centre square (**centre is always the last / 5th photo**). Two repeats
  (`r1`, `r2`) per condition.
- At the capture magnification a large square is ~**1150 px** across in the
  2048×1536 image, constant in size but at an arbitrary **position and rotation**
  (hand-placed slide), typically within ±15°.
- Beads are near-black, compact (~5–7 px core); the grid lines are faint grey and
  thin; the background is bright and slowly varying.

## 3. Input layout

Two layouts are supported and auto-detected per condition; they may be mixed in
one run.

**A. Loose photos (recommended)** — all photos dropped directly in the condition
folder:

```
<conditions root>/
  <condition A>/   10 photos, in capture order
  <condition B>/   10 photos, in capture order
```

Photos are ordered by capture time (file mtime) and split **first 5 = r1,
remainder = r2**; the centre square must be the last photo of each repeat (the
5th and 10th). If a folder does not hold exactly 10 photos the condition is still
counted and a warning is printed (e.g. `9 photos -> r1=5, r2=4`); the overall
mean pools every square, so an extra/missing shot cannot distort beads/mL — only
the per-repeat R1/R2 averages reflect the uneven split.

**B. `r1`/`r2` subfolders (original)** — unchanged:

```
<conditions root>/
  <condition A>/
    r1/  5 photos: 4 corner squares, then the CENTRE square last
    r2/  5 photos: same order
```

Any number of conditions is supported; each is auto-discovered as a folder that
either contains an `r1`/`r2` subfolder (layout B) or directly contains photos
(layout A). The conditions root is chosen at run time via a folder-picker dialog
(or passed as a command-line argument).

## 4. Component architecture

Three modules, each with a single responsibility:

| Module | Responsibility |
|---|---|
| `detect_square.py` | Locate the 1 mm² counting square (position, size, rotation) and self-check it. |
| `count_beads.py` | Detect beads inside a region and render annotations / masks. Two selectable detectors (see §5.2): **`dog`** (default) and **`legacy`**. |
| `run_counts.py` | Orchestration: discover conditions, drive detection + counting per image, aggregate to averages, compute concentration & recovery, write CSV/XLSX and image outputs. |

The active detector is chosen by the `DETECTOR` switch at the top of
`count_beads.py` (`"dog"` | `"legacy"`). Both return the identical result dict, so
`run_counts.py` and the output schema are unaffected by the choice.

Dependencies: Python 3, `opencv-python`, `numpy`, `scipy` + `scikit-image` (DoG
detector), `openpyxl` (for the xlsx). No network or GPU required.

## 5. End-to-end pipeline (raw image → final count)

```
raw photo (BGR jpg)
   │
   ▼  [detect_square.best_square]  ── locate the 1 mm² square
   ├─ grayscale
   ├─ line_maps(): black-hat + long-run opening → grid-line masks (plain + triple-boundary)
   ├─ grid_angle(): rotation by projection sharpness (±15°, robust to dense grids)
   └─ size+position fit at that angle → square corners (rotated quad)
   │
   ▼  [detect_square.verify_subdivision]  ── self-check
   └─ expect 4×4 internal lines (corner) or 5×5 (centre, last photo); flag if absent
   │
   ▼  [count_beads.count_image]  ── count beads inside the square (DEFAULT: dog)
   ├─ green channel; black out everything outside the square polygon
   ├─ M = green > MASK_MAX; neutralise masked area to its median
   ├─ Difference-of-Gaussians on the inverted image (SIGMA_1, SIGMA_2)
   ├─ prominence maxima (skimage h_maxima, 8-conn) → one point per bead
   ├─ intensity gate: keep points where the σ=1 smoothed image < GATE_FRAC*background (drops grid lines)
   └─ sub-pixel centre refinement (Hessian); count = number of points
   │
   ▼  outputs
   ├─ count per 1 mm² square (plus raw_maxima before the gate)
   ├─ unmasked, masked, and annotated demo images
   └─ [run_counts] averages → beads/mL → total beads → recovery %
```

### 5.1 Square detection (`detect_square.py`)

1. **Grid-line maps** (`line_maps`): a black-hat transform (`closing − image`,
   `BLACKHAT_K=31`) responds to anything darker than its local surroundings
   (beads *and* grid lines). Thresholding (`LINE_THRESH=12`) then a long-run
   morphological opening (`OPEN_LEN=60`, horizontal and vertical) keeps only the
   long thin structures — the grid lines — dropping the compact beads. A second
   pass (close-then-open perpendicular to the line) isolates the **triple boundary
   lines** that delimit each large square.
2. **Rotation** (`grid_angle`): for each candidate angle in `ANGLE_RANGE`
   (−15°…+15°, 0.5° steps) the line map is de-rotated and its row/column
   projections computed. At the true angle all lines fall into the same rows/cols,
   so the projections are sharply peaked (**maximum variance**). This is robust
   even in the dense centre square, where an edge-overlap fit alone can lock onto
   a wrong angle.
3. **Size + position fit** (`best_square`): at the chosen angle (± a 1° window) a
   fixed-size square (`S_RANGE` ≈ 1120–1200 px) is slid over the projections; the
   best position maximises how much its four edges overlap grid lines, with the
   triple-boundary evidence up-weighted (`BOUNDARY_W=2.5`) so a boundary anchors an
   edge while the known size completes the square. Output: the four **corners** as
   a rotated quad, plus the axis-aligned box and angle.
4. **Self-check** (`verify_subdivision`): samples the line projections at the
   expected internal divisions — ¼/½/¾ for a corner (4×4) or ⅕…⅘ for the centre
   (5×5). Returns pass/fraction; a failure is surfaced in the console and the
   `selfcheck_ok` CSV column for manual review rather than silently miscounting.

### 5.2 Bead detection (`count_beads.py`)

`count_image(path, region)` dispatches on the `DETECTOR` switch and returns the
same result dict either way. The default is **`dog`**; **`legacy`** is retained.

#### 5.2.1 DoG detector — `DETECTOR = "dog"` (default)

A port of the validated Fiji/ImageJ macro `count_beads_masked_batch.ijm`
(Difference-of-Gaussians + "Find Maxima" by prominence). Its defining property:
**each bead is one intensity peak**, so two touching beads produce two peaks and
count as two — with no area measurement and no single↔couple threshold to tune.
This is what the area-based legacy method could not do reliably.

1. **Channel & mask**: take the **green** channel as float. Everything outside the
   detected square polygon is blacked out, so the counting mask `M = green >
   MASK_MAX (20)` is exactly the square interior (dark aggregate cores ≤ 20 fall
   out of `M`, as in Fiji). The masked area is then **neutralised** to the median
   of `green[M]` so its edges create no spurious peaks.
2. **Gate image**: a σ=1 (`SIGMA_GATE`) Gaussian blur of the neutralised green,
   used later to reject grid lines.
3. **DoG**: on the **inverted** neutralised green (beads are dark),
   `dog = gaussian(inv, SIGMA_1=1.4) − gaussian(inv, SIGMA_2=2.8)` — a band-pass
   matched to the ~5.5 px bead core.
4. **Prominence maxima**: `skimage.morphology.h_maxima(dog, PROMINENCE=2.0)` with
   8-connectivity accepts a peak only if every path to any higher peak drops by
   more than `PROMINENCE`. Each maximum region yields **one point** (the pixel
   nearest its centroid — ImageJ's plateau rule). `raw_maxima` = points at this
   stage. A fixed-window max filter is deliberately **not** used: window
   suppression merges touching beads and undercounts chains.
5. **Intensity gate**: keep only points where `gate_img < GATE_FRAC * background`
   (`GATE_FRAC = 0.65`, background = per-photo median green). The threshold is
   RELATIVE, not a fixed value: bead centres and faint grid lines form two
   separate intensity peaks (beads ~0.4–0.6·bg, grid lines ~0.7·bg) and the
   cut sits in the valley between them, so it keeps the beads and drops the grid
   lines while adapting to each photo's lighting. The ROI is also eroded inward
   (`ROI_ERODE_K = 16`) beforehand so the square's own border line is not counted.
   `count` = points after the gate. (Previously a fixed `GATE_MAX = 72`, which sat
   in the middle of the bead peak and discarded ~half of the real beads.)
6. **Sub-pixel refinement** (TrackMate-style): from the 3×3 DoG neighbourhood,
   shift each point by `δ = −H⁻¹∇`, applied only if the Hessian is
   negative-definite and `|δ| ≤ 0.5 px`. This refines **positions only** and never
   changes the count.
7. **Rendering**: a red circle per counted point, drawn on the **masked** image.
   There are no clumps, so `singles = count` and the aggregate/ignored fields are 0
   and `bead_area` is blank in the result dict (schema unchanged; see §6).

#### 5.2.2 Legacy detector — `DETECTOR = "legacy"`

The original area-based method, kept for comparison. Beads are segmented by a
black-hat + high threshold (`threshold_beads`), connected components give per-blob
area/shape (`blob_stats`), only blobs whose centroid is inside the square are kept,
the single-bead area is the **mode** of in-square blob areas × `BEAD_AREA_SCALE`
(`calibrate_bead_area`), and each blob is split by `ratio = area / bead_area`:
`ratio ≤ SINGLE_MAX (1.8)` → 1 bead (green circle); `≤ AGG_CAP (10)` →
round(ratio) beads (blue box+number); above → ignored (red X). Its weakness is
that premise — separating "one large bead" from "two touching beads" by area is
inherently ambiguous, which is why `dog` is now the default.

### 5.3 Aggregation, concentration & recovery (`run_counts.py`)

- Per condition: the 5 per-square counts of `r1` and `r2` give per-repeat averages
  and an overall mean per square.
- **Concentration**: `beads_per_mL = mean_per_square / VOL_PER_SQUARE_ML`
  (`VOL_PER_SQUARE_ML = 1×10⁻⁴`).
- **Total beads**: `beads_per_mL × sample_volume_mL`.
- **Recovery %**: `total_beads(condition) / total_beads(reference) × 100`, where
  the reference is set by `REFERENCE_CONDITION` (default `"bulk"`).
- Volumes: `CONDITION_VOLUMES` pre-fills fixed volumes (e.g. bulk = 200 mL); any
  volume not listed is left **blank and editable** in the xlsx so it can be entered
  per experiment, with `total_beads` and `recovery_pct` written as **live Excel
  formulas** that recalculate on entry.

## 6. Outputs (`<conditions root>/results/`)

| Path | Contents |
|---|---|
| `results_summary.xlsx` | Per-condition averages, beads/mL, editable volume cell (yellow), and live `total_beads` / `recovery_pct` formulas. |
| `results_summary.csv` | Same table as static values (recovery blank where volume is entered in the xlsx). |
| `results_per_image.csv` | Per photo: count, raw maxima (pre-gate; `dog` only, blank for `legacy`), singles, aggregate beads/clumps, ignored, calibrated bead area, detect score, self-check pass/fraction. Column schema is the same for both detectors; under `dog` the singles column equals the count and the aggregate/ignored/bead-area columns are 0/blank (no clumps). |
| `<condition>/unmasked/<repeat>/` | Raw original photos. |
| `<condition>/masked/<repeat>/` | Original with only the counting square visible. |
| `<condition>/annotated/<repeat>/` | Masked image with counted beads drawn. `dog`: one red circle per bead. `legacy`: green = single, blue = 2–10 clump, red X = ignored. |
| `all_annotated/` | Every annotated image gathered into one flat folder, renamed `<condition>__<repeat>__pNN__<original>.jpg` so they sort by condition, then repeat, then photo order — for quick scrolling. |

## 7. Tunable parameters

`count_beads.py`
- `DETECTOR` (`"dog"`) — which detector to run: `"dog"` (default) or `"legacy"`.

DoG detector (`dog`):
- `PROMINENCE` (2.0) — minimum peak prominence; lower detects fainter beads (risks
  noise), higher is more conservative. The main sensitivity knob.
- `GATE_FRAC` (0.65) — a peak counts only if the σ=1 smoothed image is below
  `GATE_FRAC * background`; sits in the valley between the dark bead-centre peak
  and the brighter grid-line peak. Lower = stricter (drops faint beads), higher
  risks admitting grid lines. (Replaced a fixed `GATE_MAX = 72`.)
- `ROI_ERODE_K` (16) — erode the detected square inward by ~half this many px
  before detecting, so its own border line is not counted as beads.
- `SIGMA_1` (1.4), `SIGMA_2` (2.8) — DoG band-pass, matched to the bead core;
  **update if the bead size in pixels changes** (e.g. different magnification).
- `MASK_MAX` (20), `SIGMA_GATE` (1.0) — background mask threshold and gate blur.

Legacy detector (`legacy`):
- `BLACKHAT_THRESH` (45) — bead vs grid-line separation; higher drops faint lines
  and shrinks beads.
- `BEAD_AREA_SCALE` (1.6) — single-bead size bias; higher = more conservative clump
  counting. ~1.6–1.8 for sparse samples, ~1.2–1.4 for very dense samples.
- `SINGLE_MAX` (1.8) — single↔couple boundary. `AGG_CAP` (10) — max clump size
  counted before a blob is ignored.
- `MIN_BLOB_PX` (8), `NOISE_FRAC` (0.40) — noise floors.

`detect_square.py`
- `S_RANGE` (~1120–1200 px) — square size at the capture magnification; **update if
  the magnification changes.**
- `ANGLE_RANGE` (±15°) — rotation search. `BOUNDARY_W` (2.5) — weight of
  triple-boundary evidence.

`run_counts.py`
- `CONDITION_VOLUMES`, `REFERENCE_CONDITION`, `VOL_PER_SQUARE_ML`.

## 8. Assumptions, limitations & failure modes

- **Fixed square size**: detection assumes the ~1150 px square size; a different
  magnification requires updating `S_RANGE`. The self-check flags gross failures.
- **Photo ordering**: the centre square must be the last of the 5 photos so the
  self-check expects the right subdivision.
- **Detector-specific — `dog`**: touching beads are separated by peak, not area, so
  there is no single↔couple threshold and no clump cap; the trade-off moves to
  `PROMINENCE` (too low invents peaks in noise, too high merges/loses faint beads)
  and `GATE_FRAC` (grid-line rejection). Very dense clumps where individual cores are
  no longer resolvable will still under-count. `SIGMA_1/2` assume the current bead
  size in pixels and must be updated if magnification changes.
- **Detector-specific — `legacy`**: counting clumps by area cannot perfectly
  separate "one large bead" from "two touching beads"; `BEAD_AREA_SCALE` and the
  aggregate cap tune the trade-off, and clumps above the cap are discarded (logged
  as `ignored`), so dense counts are mildly conservative. This ambiguity is the
  reason `dog` is the default.
- **Global parameters**: one parameter set is applied to all images in a run
  (`PROMINENCE`/`GATE_FRAC` for `dog`, `BEAD_AREA_SCALE` for `legacy`); mixing sparse
  and very dense conditions in one run is a compromise for both.
- **Self-check is structural**, not exact: it confirms the expected internal grid
  is present, not sub-pixel edge accuracy. The masked/annotated images are the
  final visual check.

## 9. How to run

```
python run_counts.py                 # opens a folder picker
python run_counts.py "D:\path\to\conditions"   # or pass the folder directly
```
See `README.md` for the operator-facing quick start.
