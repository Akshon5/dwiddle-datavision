# Dwiddle (DataVision): AI-Powered Drawing Riddle Challenge

**Course:** DataVision / Edge AI & Computer Vision  
**Project:** Dwiddle — The Neural Vision Challenge  
**Dataset:** Google Quick, Draw! (28×28 Grayscale Bitmaps, CC BY 4.0)  
**Framework:** PyTorch & Streamlit  

---

## 1. Project Overview & Concept

**Dwiddle** is an interactive AI-powered drawing recognition and riddle challenge powered by custom Convolutional Neural Networks (CNN) trained on Google Quick, Draw! sketches.

Players are presented with clever riddles across three difficulty levels (**Easy**, **Medium**, and **Hard**). Instead of typing answers, players must **sketch their solution on an interactive canvas**. In real-time, the drawing is preprocessed, centered, and passed through a neural network that predicts the object class and confidence score.

### Key Highlights:
- **Interactive Gameplay:** 15 riddles across 3 tiers (5 Easy, 5 Medium, 5 Hard) in the expanded 20-class mode, plus an 8-chamber Facility Escape survival mode.
- **Real-Time CNN Inference:** Custom PyTorch CNN running on CPU in under **0.5 ms per sketch**.
- **Robust Canvas Preprocessing:** Automatic background polarity detection, stroke thresholding, bounding-box cropping, 18% margin square centering, and bilinear downsampling to $28 \times 28$.
- **High Accuracy:** **94.22% test accuracy** (0.9419 Macro-F1) on the 20-class model and **95.11% test accuracy** on the 8-class model.
- **Game Engine & Mechanics:** Stateful life system (3 lives), hints, first-attempt bonuses (+50 pts), base riddle solve points (+100 pts), remaining lives bonus (+25 pts/life), and blank-canvas protection.

---

## 2. Supported Object Classes

### Expanded 20-Class Vocabulary
The primary model recognizes 20 common objects:
`airplane`, `apple`, `axe`, `bicycle`, `book`, `candle`, `chair`, `clock`, `cup`, `door`, `fish`, `flower`, `house`, `key`, `parachute`, `scissors`, `shoe`, `star`, `tree`, `umbrella`.

### 8-Class Survival Vocabulary
The original survival mode recognizes:
`airplane`, `apple`, `axe`, `bicycle`, `candle`, `door`, `key`, `parachute`.

---

## 3. Game Mechanics & Rules

- **Lives:** Starting with **3 lives** (`❤️❤️❤️`).
- **Attempts & Penalties:** Submitting an incorrect object or falling below the confidence threshold (default 40%) costs 1 life.
- **Blank Canvas Protection:** Submitting an empty or near-empty canvas (<12 stroke pixels) is rejected safely without costing a life.
- **Hints:** A single hint is available for each riddle via an expander.
- **Scoring System:**
  - **Riddle Solved:** `+100 points`
  - **First-Attempt Bonus:** `+50 bonus points`
  - **Remaining Lives Bonus:** `+25 bonus points` per remaining life upon victory.
  - **Max Score:** $5 \times 150 + 3 \times 25 = \mathbf{825\text{ points}}$ per 5-riddle run.

---

## 4. CNN Architecture

The model uses a clean, efficient Convolutional Neural Network architecture (`QuickDrawCNN`):

```
Input: (Batch, 1, 28, 28) [float32 normalized in 0.0 - 1.0]
  │
  ├── Conv2d(1 -> 32, kernel=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2x2)  --> (B, 32, 14, 14)
  ├── Conv2d(32 -> 64, kernel=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2x2) --> (B, 64, 7, 7)
  ├── Conv2d(64 -> 128, kernel=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2x2) --> (B, 128, 3, 3)
  │
  └── Fully Connected Head:
        Flatten (1,152 features)
        Dropout(p=0.3)
        Linear(1152 -> 128) + ReLU
        Dropout(p=0.2)
        Linear(128 -> num_classes) [8 or 20 logits]
```

- **Parameters:** ~243K parameters.
- **Inference Time:** ~0.4 ms on standard CPU.
- **Checkpoints:**
  - `models/expanded_cnn_20.pt` (20 classes, 94.22% accuracy)
  - `models/baseline_cnn.pt` (8 classes, 95.11% accuracy)

---

## 5. Preprocessing Pipeline Contract

The preprocessing module (`src/preprocess.py`) standardizes user sketches:

1. **Polarity Normalization:** Inspects canvas corners to detect background polarity. Inverts light background / dark ink into black background ($0.0$) and light strokes ($> 0.0$).
2. **Stroke Filtering:** Enforces minimum stroke thresholds (rejects empty clicks with $< 12$ active pixels).
3. **Bounding-Box Extraction:** Finds the tight bounding box containing all drawn strokes.
4. **Square Padding & Centering:** Adds an 18% margin and centers the sketch inside a square aspect ratio.
5. **Bilinear Downsampling:** Resamples the drawing to canonical $28 \times 28$ resolution.
6. **Normalization:** Scales pixel intensities to $[0.0, 1.0]$ float32 PyTorch tensor of shape `(1, 1, 28, 28)`.

---

## 6. Project Directory Structure

```
dwiddle-datavision/
├── app.py                                  # Streamlit interactive application
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── .gitignore                              # Git exclusion rules
├── .streamlit/
│   └── config.toml                         # Streamlit dark theme & server configuration
├── src/
│   ├── model.py                            # QuickDrawCNN PyTorch architecture
│   ├── preprocess.py                       # Canvas preprocessor & normalizer
│   ├── inference.py                        # QuickDrawPredictor inference wrapper
│   ├── game_engine.py                      # Game state machine, scoring, & lives logic
│   ├── challenges.py                       # Riddle definitions & target acceptance
│   ├── dataset.py                          # Quick, Draw! data downloader & split builder
│   ├── dataset_expanded.py                 # 20-class dataset builder
│   ├── train.py                            # 8-class training script
│   ├── train_expanded.py                   # 20-class training script
│   ├── evaluate.py                         # 8-class test evaluation & latency
│   ├── evaluate_expanded.py                # 20-class test evaluation & latency
│   ├── validate_sketches.py                # Canvas validation benchmark
│   └── fresh_sketch_evaluation.py          # Empirical vector sketch benchmark
├── tests/
│   ├── test_game.py                        # Core game engine unit tests
│   ├── test_playthrough.py                 # Full game playthrough tests
│   ├── test_canvas_integration.py          # Preprocessor and canvas integration tests
│   ├── test_game_state_exhaustive.py       # State boundary & reset tests
│   ├── test_expanded_model.py              # 20-class inference & model tests
│   └── test_expanded_gameplay.py           # 20-class game engine & riddle tests
├── models/
│   ├── baseline_cnn.pt                     # 8-class PyTorch model weights
│   ├── label_mapping.json                  # 8-class label mapping dictionary
│   ├── expanded_cnn_20.pt                  # 20-class PyTorch model weights
│   └── label_mapping_20.json               # 20-class label mapping dictionary
├── data/
│   ├── train.npz                           # 8-class training split
│   ├── val.npz                             # 8-class validation split
│   ├── test.npz                            # 8-class held-out test split
│   ├── split_metadata.json                 # 8-class metadata
│   └── expanded_20/
│       ├── train.npz                       # 20-class training split (21,000 samples)
│       ├── val.npz                         # 20-class validation split (4,500 samples)
│       ├── test.npz                        # 20-class held-out test split (4,500 samples)
│       ├── split_metadata.json             # 20-class partition metadata
│       └── label_mapping.json              # 20-class label dictionary
├── dataset_recon/                          # Dataset exploration & reconnaissance
└── reports/                                # Training curves, confusion matrices, & logs
```

---

## 7. Setup & Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.14)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/Akshon5/dwiddle-datavision.git
cd dwiddle-datavision

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 8. Authors & Acknowledgments

- **Author:** Akshon ([@Akshon5](https://github.com/Akshon5))  
- **Email:** `achoudhary_be23@thapar.edu`  
- **Course / Project:** DataVision  
- **Dataset:** [Google Quick, Draw! Dataset](https://quickdraw.withgoogle.com/data) (Creative Commons Attribution 4.0 International License)
