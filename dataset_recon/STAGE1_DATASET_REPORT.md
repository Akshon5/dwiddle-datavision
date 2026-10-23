# STAGE 1: DATASET RECONNAISSANCE REPORT
## Project: Draw to Survive — An AI-Powered Drawing Adventure
**Course:** Edge AI and Robotics: Data Center Vision  
**Date of Inspection:** October 13, 2026  
**Status:** Reconnaissance Complete — Awaiting Approval for Stage 2  

---

## 1. Official Dataset Source and Citation

- **Dataset Name:** The Quick, Draw! Dataset
- **Publisher / Provider:** Google Creative Lab / Google, Inc.
- **Official GitHub Repository:** [https://github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- **Interactive Dataset Explorer:** [https://quickdraw.withgoogle.com/data](https://quickdraw.withgoogle.com/data)
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Official Citation:**
  > Ha, D., & Eck, D. (2018). *A Neural Representation of Sketch Drawings*. International Conference on Learning Representations (ICLR 2018). arXiv:1704.03477.
  > Google Creative Lab (2017). *Quick, Draw! Dataset*. Available at: https://github.com/googlecreativelab/quickdraw-dataset.

---

## 2. Access Method and Connectivity Verification

1. **Internet Access Check:**
   - **Status:** Verified working.
   - **Sandbox Isolation:** The default agent sandbox prevents outbound DNS resolution. With user-approved network access (`BypassSandbox: true`), direct HTTPS communication with `raw.githubusercontent.com` and `storage.googleapis.com` succeeded reliably.
2. **Official Category List:**
   - Fetched `categories.txt` directly from `raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt`.
   - **Total verified categories:** Exactly **345 categories**.
3. **Public Storage Hosting:**
   - Google Cloud Storage (GCS) Multi-Regional Public Bucket: `gs://quickdraw_dataset`
   - Public HTTPS Base URL: `https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<category>.npy`
4. **HTTP Protocol & Range Support:**
   - Server returns `Accept-Ranges: bytes`.
   - Dynamic HTTP Range queries returned status code `HTTP/2 206 Partial Content`.
   - Validated server honors range headers with exact byte matching via `Content-Range: bytes <start>-<end>/<total>`.

---

## 3. Verified Candidate Categories

All 16 candidate categories provided in the task brief were inspected against the official `categories.txt` and verified on Google Cloud Storage.

| Candidate Category | Official Name in Dataset | Status on GCS | Total Dataset Examples | Full `.npy` Size (MB) | Verified Header Offset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **key** | `key` | Available | 160,965 | 120.35 MB | 80 bytes |
| **hammer** | `hammer` | Available | 119,012 | 88.98 MB | 80 bytes |
| **axe** | `axe` | Available | 124,122 | 92.80 MB | 80 bytes |
| **ladder** | `ladder` | Available | 125,389 | 93.75 MB | 80 bytes |
| **bridge** | `bridge` | Available | 133,010 | 99.45 MB | 80 bytes |
| **flashlight** | `flashlight` | Available | 239,763 | 179.27 MB | 80 bytes |
| **door** | `door` | Available | 120,230 | 89.89 MB | 80 bytes |
| **airplane** | `airplane` | Available | 151,623 | 113.37 MB | 80 bytes |
| **parachute** | `parachute` | Available | 127,319 | 95.19 MB | 80 bytes |
| **screwdriver** | `screwdriver` | Available | 116,313 | 86.97 MB | 80 bytes |
| **candle** | `candle` | Available | 141,545 | 105.83 MB | 80 bytes |
| **lantern** | `lantern` | Available | 149,912 | 112.09 MB | 80 bytes |
| **shoe** | `shoe` | Available | 120,231 | 89.89 MB | 80 bytes |
| **apple** | `apple` | Available | 144,722 | 108.21 MB | 80 bytes |
| **cup** | `cup` | Available | 130,721 | 97.74 MB | 80 bytes |
| **bicycle** | `bicycle` | Available | 126,527 | 94.60 MB | 80 bytes |

**Summary:** 100% of the 16 candidate categories exist as exact matches with large sample counts (>116,000 drawings each). Together, these 16 categories contain **2,231,434 drawings** (~1.75 GB).

---

## 4. Available Formats and File-Size Estimates

The Quick, Draw! dataset is provided in four primary formats across the 345 categories:

1. **Raw Vector Drawings (`.ndjson`)**:
   - Location: `gs://quickdraw_dataset/full/raw/`
   - Description: Full timestamped stroke arrays `[[x, y, t], ...]`, country code, recognized status, and drawing id.
   - Size: ~1 GB to 2.5 GB per category (~500 GB total).
   - Practicality for 2-day deadline: Unnecessarily complex; requires rasterization pipelines.
2. **Simplified Vector Drawings (`.ndjson`)**:
   - Location: `gs://quickdraw_dataset/full/simplified/`
   - Description: Ramer–Douglas–Peucker simplified vectors scaled to 256×256 canvas with stroke timing removed.
   - Size: ~200 MB to 600 MB per category.
3. **Binary Compressed Vectors (`.bin`)**:
   - Location: `gs://quickdraw_dataset/full/binary/`
   - Description: Custom packed binary stroke format for fast loading in NodeJS/Python.
   - Size: ~50 MB to 150 MB per category.
4. **Pre-rendered Grayscale Bitmaps (`.npy`) — RECOMMENDED & SELECTED**:
   - Location: `gs://quickdraw_dataset/full/numpy_bitmap/`
   - Description: Rendered 28×28 grayscale bitmaps centered in canvas, stored as flat row-major arrays of 784 bytes per drawing.
   - Size: ~85 MB to 180 MB per category.
   - Header: 80 bytes uncompressed NumPy v1.0 header (`{'descr': '|u1', 'fortran_order': False, 'shape': (N, 784)}`).
   - Partial Range Access: Supported natively by Google Cloud Storage.

---

## 5. Efficient Data Acquisition Method (HTTP Range Streaming)

### Why Partial Downloading Succeeded
- In NumPy `.npy` (v1.0), the file layout is uncompressed and strictly contiguous:
  $$\text{Byte Range for drawing } i = [\text{Header Offset} + i \times 784, \; \text{Header Offset} + (i+1) \times 784 - 1]$$
- Dynamic header parsing was implemented:
  1. Fetch bytes `0–255` via HTTP Range.
  2. Verify HTTP `206 Partial Content` and `Content-Range`.
  3. Verify magic bytes `\x93NUMPY` and version `1.0`.
  4. Dynamically compute header length: `hdr_len = int.from_bytes(data[8:10], 'little')`, resulting in `total_hdr_offset = 80` bytes.
  5. Validate `shape == (N, 784)` and `dtype == '|u1'` (uint8).
  6. Request exactly $K \times 784$ bytes starting at offset 80.
  7. Validate received payload byte count before reshaping into `(K, 28, 28)`.

### Bandwidth & Storage Comparison
- **Downloading full files for 16 categories:** $1,749.5\text{ MB}$ (~1.75 GB)
- **Downloading 1,000 samples per category via Range requests:** $16 \times 765.75\text{ KB} = 12.25\text{ MB}$
- **Savings:** **99.3% reduction in bandwidth and local storage**, with zero risk of running out of disk space.

---

## 6. Sample Counts, Image Properties, and Quality Inspection

### Verified Image Properties
- **Array Shape:** `(1000, 28, 28)` per sampled category
- **Data Type:** `uint8` (`0` to `255`)
- **Pixel Value Conventions:**
  - `0`: Background (black)
  - `1..255`: Anti-aliased stroke intensity (white lines)
  - *Note for UI canvas integration:* Web/Streamlit drawing canvases typically produce black strokes on white canvas (255 background, 0 stroke). Preprocessing in inference must invert strokes (`255 - canvas_pixels`) or scale appropriately.

### Quality and Anomaly Findings
Across the 16,000 sampled drawings:
1. **Completely Blank Drawings:** Exactly **0** found across all 16 sampled sets.
2. **Minimal / Sparse Drawings (< 15 non-zero pixels):** Very few (<0.1%). These are typically abortive sketches where the user drew a single dot, dash, or gave up.
3. **Overly Dense / Scribbled Drawings (> 450 non-zero pixels):** Occasional (1–2 per 1,000). These represent rage-scribbles, full-box shading, or heavily textured patterns.
4. **Average Stroke Pixel Count:**
   - Lightest: `screwdriver` (140.1 pixels) and `candle` (152.1 pixels)
   - Densest: `lantern` (251.3 pixels) and `parachute` (243.1 pixels)

---

## 7. Recommended 8 Categories for the Game

We recommend selecting the following **8 categories** for "Draw to Survive":

| Category | Primary Visual Silhouette | Gameplay / Adventure Mechanic | Why Selected |
| :--- | :--- | :--- | :--- |
| **1. apple** | Rounded body + top stem & leaf | Hunger / Healing / Foraging | Circular geometry, highly distinct from tools and vehicles. |
| **2. airplane** | Cruciform fuselage + swept wings | Aerial Escape / Rescue Signal | Strong horizontal/vertical axis symmetry; unmistakable silhouette. |
| **3. bicycle** | Dual separated wheels + frame | Ground Transport / Fast Travel | Distinct topological feature (two circular wheels connected by lines). |
| **4. candle** | Vertical cylinder + teardrop flame | Dark Cave / Dungeon Illumination | Simple vertical stroke with isolated flame; distinct from tools. |
| **5. door** | Rectangular frame + knob / panels | Barrier / Room Transition / Exit | Clean geometric perimeter; essential escape-room mechanic. |
| **6. key** | Circular bow + long shaft + teeth | Unlock Chests / Prison Escape | Unmistakable aspect ratio and notched teeth; core puzzle element. |
| **7. parachute** | Hemispherical dome + angled cords | Chasm Jump / Fall Survival | Wide curved dome with converging triangular cords; distinct from all others. |
| **8. axe** | Handle + wedge-shaped blade | Combat / Chop Wood / Clear Path | Strong, distinct angled head on a stick. Chosen as the single representative tool. |

### Why This Set is Optimal
1. **Visual Orthogonality:** Each category occupies a distinct structural primitive:
   - Circle: `apple`
   - Concentric dual-rings: `bicycle`
   - Cross: `airplane`
   - Rectangle: `door`
   - Teardrop + cylinder: `candle`
   - Notched rod: `key`
   - Dome + canopy strings: `parachute`
   - Wedge + rod: `axe`
2. **CNN Separability:** A small CNN (e.g. 2–3 convolutional layers with 32–64 filters) can easily learn these structural features without confusions.
3. **Game Narrative Versatility:** Allows designing rich adventure challenges (e.g., "The path is blocked by vines $\rightarrow$ draw an **axe**; you enter a pitch-black tomb $\rightarrow$ draw a **candle**; you reach a cliff edge $\rightarrow$ draw a **parachute**; you find an ancient gate $\rightarrow$ draw a **key**").

---

## 8. Categories to Avoid (and Why)

| Category | Reason to Avoid |
| :--- | :--- |
| **hammer** | **High Confusion Risk with Axe & Screwdriver:** In fast sketches, players draw a handle with an indistinct crossbar or head. A small CNN struggles to differentiate a hammer head from an axe blade or mallet. |
| **screwdriver** | **Ambiguous Silhouette:** A narrow cylinder with a handle resembles a wand, pencil, key stem, candle, or knife. |
| **flashlight** | **Redundant with Candle:** Can be drawn as a vertical cylinder or a horizontal cone, frequently confused with candle, microphone, or hammer. |
| **lantern** | **Excessive Intra-Class Variance:** Inspection revealed wildly varying sketches: camping lanterns, street lamps, paper lanterns, and bags. High stroke complexity makes classification noisy. |
| **ladder** | **High Confusion with Bridge & Door:** Parallel vertical lines with crossbars are frequently confused with doorways, suspension bridges, or train tracks. |
| **bridge** | **Excessive Structural Variance:** Drawings include beam bridges, arched stone bridges, suspension bridges with cables, and simple planks. Too diverse for a beginner-level CNN. |
| **shoe** | **Style Inconsistency:** Spans sneakers, high heels, boots, and flat slippers; players draw them at different orientations. |
| **cup** | **Backup Only:** While usable, sketches without handles look like simple cylinders, planters, or wastebaskets. Kept as reserve. |

---

## 9. Proposed Plan for Stage 2 (Data Preparation & Partitioning)

For Stage 2, we propose building a balanced dataset from the 8 recommended classes:

1. **Target Dataset Size:**
   - **1,500 drawings per category** (Total = $8 \times 1,500 = 12,000$ drawings).
   - *Rationale:* 12,000 images is lightweight (only 9.4 MB total), trains a small CNN in < 1 minute on laptop CPU/GPU, yet provides enough statistical density for >90% validation accuracy.
2. **Quality Cleaning & Outlier Filtering:**
   - Discard drawings with stroke pixel count $< 15$ (empty / aborted drawings).
   - Discard drawings with stroke pixel count $> 450$ (solid scribbles / noise).
   - Shuffle with a fixed random seed (`seed=42`) for reproducibility.
3. **Partition Strategy:**
   - **Train Set (70%):** 1,050 drawings per category (8,400 total)
   - **Validation Set (15%):** 225 drawings per category (1,800 total)
   - **Test Set (15%):** 225 drawings per category (1,800 total)
4. **Normalization & Augmentation:**
   - Scale pixel values to $[0.0, 1.0]$.
   - Slight random rotation ($\pm 10^\circ$) and translation ($\pm 2$ pixels) during CNN training to accommodate varied player drawing styles.
   - Reshape to `(N, 28, 28, 1)` for 2D CNN input.

---

## 10. Exact Files Downloaded and Storage Footprint

All reconnaissance files are organized in the project directory:

```
/Users/akshon/Desktop/datavision/
├── dataset_recon/
│   ├── categories.txt                     (2.8 KB, official 345 categories)
│   ├── category_metadata.json             (4.2 KB, verified GCS metadata)
│   ├── category_inventory.csv             (3.5 KB, machine-readable inventory)
│   ├── download_and_inspect.py            (10.5 KB, robust acquisition script)
│   ├── samples/                           (16 files, 12.25 MB total)
│   │   ├── airplane_1000.npy              (765.8 KB, 1,000 drawings)
│   │   ├── apple_1000.npy                 (765.8 KB, 1,000 drawings)
│   │   ├── axe_1000.npy                   (765.8 KB, 1,000 drawings)
│   │   ├── bicycle_1000.npy               (765.8 KB, 1,000 drawings)
│   │   ├── bridge_1000.npy                (765.8 KB, 1,000 drawings)
│   │   ├── candle_1000.npy                (765.8 KB, 1,000 drawings)
│   │   ├── cup_1000.npy                   (765.8 KB, 1,000 drawings)
│   │   ├── door_1000.npy                  (765.8 KB, 1,000 drawings)
│   │   ├── flashlight_1000.npy            (765.8 KB, 1,000 drawings)
│   │   ├── hammer_1000.npy                (765.8 KB, 1,000 drawings)
│   │   ├── key_1000.npy                   (765.8 KB, 1,000 drawings)
│   │   ├── ladder_1000.npy                (765.8 KB, 1,000 drawings)
│   │   ├── lantern_1000.npy               (765.8 KB, 1,000 drawings)
│   │   ├── parachute_1000.npy             (765.8 KB, 1,000 drawings)
│   │   ├── screwdriver_1000.npy           (765.8 KB, 1,000 drawings)
│   │   └── shoe_1000.npy                  (765.8 KB, 1,000 drawings)
│   └── gallery/
│       ├── gallery_recommended_8_inverted.png (34 KB, 8 classes x 10 samples, white bg)
│       ├── gallery_recommended_8_original.png (34 KB, 8 classes x 10 samples, black bg)
│       ├── gallery_all_candidates.png         (46 KB, 16 classes x 8 samples)
│       └── gallery_quality_inspection.png     (10 KB, sparse vs dense anomaly sheet)
├── STAGE1_DATASET_REPORT.md               (This report)
└── .venv/                                 (Project virtual environment)
```

- **Total storage used by downloaded data samples:** **12.25 MB**
- **Total storage of visual galleries:** **124 KB**
- **Large files downloaded:** **None**. No full category files were downloaded; all files were strictly sampled via HTTP Range requests.

---

## 11. Unresolved Questions and Limitations

1. **Stroke Order vs. Static Image:** The `.npy` format contains only static 28×28 rasterized sketches, losing stroke timing and sequence order. For a CNN classifier, this is desirable and standard. If recurrent models (Sketch-RNN / LSTM) were desired, the vector formats (`.ndjson`) would be needed. For a 2-day deadline, a 2D CNN on 28×28 bitmaps is much simpler, faster, and reliable.
2. **Inverted Colors on Interactive Canvas:** Quick, Draw! bitmaps store strokes as white pixels on black background (`0` background, `255` stroke). Interactive web canvases typically draw black lines on a white background. The inference preprocessor must invert or match intensities.
3. **Resolution Constraint (28×28):** 28×28 is sufficient for simple, distinct shapes like keys, apples, and bicycles, but fine details (like the individual teeth on a key or the wick of a candle) are blurred across a few pixels. This reinforces the need to avoid visually confusable classes like `hammer` vs `axe`.
