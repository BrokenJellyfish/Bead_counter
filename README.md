# Bead counting

Counts black particles per **1 mm² counting square** in hemocytometer photos,
handles aggregates, restricts counting to the detected square, and scales up to a
concentration and total bead count for the sample.

*For the algorithm, full parameter list, and failure modes, see `bead_counter_SPEC.md`.*

## Requirements
Python 3 plus a few libraries. Install them in one line, either way:

- **conda / Anaconda:** `conda env create -f environment.yml` then `conda activate bead-counter`
- **pip:** `pip install -r requirements.txt`

(Libraries used: `opencv-python`, `numpy`, `scipy`, `scikit-image`, `openpyxl`.)

## Run it
1. Open a terminal in this folder (File Explorer → address bar → type `powershell` → Enter).
2. Run:
   ```
   python run_counts.py
   ```
3. A folder-picker window opens — select the folder that contains your condition
   folders. It can be anywhere and different every time.
4. Results are written to a `results/` folder **inside the folder you picked**.

To skip the dialog (e.g. repeat runs), pass the path directly:
```
python run_counts.py "D:\path\to\conditions folder"
```

## Input layout
Each condition is its own folder. **Two layouts are supported** — use whichever is
convenient; both are detected automatically and can be mixed in the same run:

**A. Loose photos (simplest, recommended)** — just drop all 10 photos into the
condition folder:
```
<condition name>/
└── 10 photos (5 for repeat 1, then 5 for repeat 2)
```
- Photos are ordered by **capture time**, then split **first 5 = R1, rest = R2**.
- The centre square (finely subdivided, 5×5) should be the **last** of each repeat's
  5 photos (i.e. the 5th and the 10th); the others are corner 4×4 squares.
- **Took one more/one less photo by mistake?** No problem — the folder is still
  counted and a `WARNING` is printed (e.g. `9 photos -> r1=5, r2=4`). The overall
  average pools every photo, so an extra or missing shot can't distort beads/mL;
  only the per-repeat R1/R2 averages reflect the uneven split.

**B. r1/r2 subfolders (original)** — still works unchanged:
```
<condition name>/
├── r1/   5 photos: 4 corner squares, then the CENTRE square LAST
└── r2/   5 photos: same order
```

- Any number of `<condition>` folders is fine; all are found automatically.
- Chamber assumed: Neubauer improved, large square = 1 mm² = 1e-4 mL, imaged at the
  magnification where the square is ~1150 px. If the zoom changes, update `S_RANGE`
  in `detect_square.py`; the self-check column will flag bad detections.

## Outputs (`results/`)
- `results_summary.xlsx` — per condition: averages, beads/mL, and a **live
  recovery calculation**. The `sample_volume_mL` column is highlighted yellow and
  editable; `total_beads` and `recovery_pct` are Excel formulas. Type the
  recovered volume into the (blank) transferred row and recovery % updates itself.
  Set which condition is the 100% reference via `REFERENCE_CONDITION`, and any
  fixed volumes to pre-fill via `CONDITION_VOLUMES`, at the top of `run_counts.py`.
- `results_summary.csv` — same table as static values (recovery blank where the
  volume is left to be entered in the xlsx).
- `results_per_image.csv` — per photo: count, singles, aggregate beads/clumps,
  ignored aggregates, calibrated bead size, detect score, self-check pass/fraction.
- `<condition>/unmasked/<repeat>/` — raw original photos.
- `<condition>/masked/<repeat>/` — original with only the counting square visible.
- `<condition>/annotated/<repeat>/` — masked image with every counted bead marked
  by a **red circle** (one circle = one bead).
- `all_annotated/` — **every** annotated image gathered into one folder for easy
  scrolling, renamed `<condition>__<repeat>__pNN__<original>.jpg` so they sort by
  condition, then repeat, then photo order (e.g. `1a. bulk__r1__p01__QS_5026.jpg`).

## How beads are counted (`count_beads.py`)
The default detector is `DETECTOR = "dog"` (difference-of-Gaussians + prominence
maxima): each bead is one local intensity peak, so touching beads are counted
separately without any clump/area guessing.
- A peak is kept only if it is clearly darker than the photo's background —
  `gate_img < GATE_FRAC * background` (`GATE_FRAC = 0.65`). This threshold is
  **relative to each photo's own background** and sits in the valley between the
  dark bead peak and the brighter grid-line peak, so it keeps beads and rejects
  grid lines across different lighting. *(This replaced a fixed cutoff of 72 that
  sat in the middle of the bead population and discarded ~half the real beads.)*
- The counting square is eroded inward by a few px (`ROI_ERODE_K`) first, so the
  square's own border line isn't mistaken for a ring of beads.
- `PROMINENCE` is the main sensitivity knob (lower = detect fainter beads, higher =
  more conservative). See `bead_counter_SPEC.md` for the full parameter list.

(A `"legacy"` blob/area detector is also present — green circle = single, blue box =
clump, red X = ignored clump — but `"dog"` is the one used.)

## Concentration
1 square = 1e-4 mL, so beads/mL = (avg beads per square) × 1e4, and total beads =
beads/mL × the sample volume. The sample volume is entered **per condition** in the
`results_summary.xlsx` (yellow cells); fixed volumes can be pre-filled via
`CONDITION_VOLUMES` in `run_counts.py` (currently `bulk = 200 mL`), and the 100 %
reference for recovery is set by `REFERENCE_CONDITION` (currently `bulk`).
