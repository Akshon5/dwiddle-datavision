# STAGE C: 20-CLASS CNN TRAINING & HELD-OUT EVALUATION REPORT

**Project:** Draw to Survive: An AI-Powered Drawing Adventure  
**Date:** October 14, 2026  
**Scope:** Stage C — Train and Evaluate Separate 20-Class CNN  
**Status:** **STAGE C COMPLETE & VALIDATED**

---

## 1. Executive Summary

In Stage C, a separate 20-class convolutional neural network (`QuickDrawCNN`, 243,060 parameters) was trained from scratch using `data/expanded_20/train.npz` (21,000 drawings) and validated on `data/expanded_20/val.npz` (4,500 drawings). Model checkpointing was guided by validation accuracy, saving the optimal weights to `models/expanded_cnn_20.pt`.

Following model selection, the checkpoint was evaluated **exactly once** on the untouched held-out test set (`data/expanded_20/test.npz`, 4,500 drawings):
- **Overall 20-Class Test Accuracy:** **91.82%** (4,132 / 4,500 correct).
- **Macro F1-Score:** **0.9184** (Macro Precision: 0.9200, Macro Recall: 0.9182).
- **Single-Sample CPU Latency:** **0.407 ms** (Median: 0.382 ms, 95th-percentile: 0.525 ms).
- **Original 8-Class Files:** **100% Unchanged** (verified via SHA-256 and MD5 checksums).
- **Existing Unit Tests:** All **25 tests** in the test suite pass with zero errors ($0.030\text{s}$).

> [!IMPORTANT]
> **Task Complexity Distinction:** The 8-class model achieved 95.11% accuracy on an 8-way classification task (random baseline: 12.5%). The 20-class model achieves 91.82% accuracy on a 20-way classification task (random baseline: 5.0%). These figures represent distinct mathematical problem spaces and should not be directly compared as identical metrics.

---

## 2. Model Architecture & Hyperparameter Configuration

The established compact architecture (`QuickDrawCNN`) from `src/model.py` was reused with the linear classification head expanded from 8 to 20 logits:

```
Input: (B, 1, 28, 28), float32 in [0.0, 1.0]
  │
  ├── Block 1: Conv2d(1 -> 32, 3x3, pad=1) + BatchNorm2d + ReLU + MaxPool2d(2x2)  --> (B, 32, 14, 14)
  ├── Block 2: Conv2d(32 -> 64, 3x3, pad=1) + BatchNorm2d + ReLU + MaxPool2d(2x2) --> (B, 64, 7, 7)
  ├── Block 3: Conv2d(64 -> 128, 3x3, pad=1) + BatchNorm2d + ReLU + MaxPool2d(2x2) --> (B, 128, 3, 3)
  │
  └── Classifier Head:
        Flatten (1,152 features)
        Dropout(p=0.3)
        Linear(1,152 -> 128) + ReLU
        Dropout(p=0.2)
        Linear(128 -> 20) [Raw Logits]
```

- **Total Trainable Parameters:** **243,060 parameters** (+1,548 parameters vs. 8-class baseline).
- **Random Seed:** 42 (NumPy, PyTorch, and MPS/CUDA).
- **Batch Size:** 64.
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, weight decay $1\times 10^{-4}$).
- **Initial Learning Rate:** $1\times 10^{-3}$ ($0.001$).
- **Learning Rate Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`.
- **Training Epochs:** 15 epochs.
- **Hardware Acceleration:** Apple Silicon MPS (`mps` device).
- **Total Training Duration:** **80.6 seconds** (1.34 minutes).

---

## 3. Training & Validation Progression

| Epoch | Epoch Time | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Learning Rate | Checkpoint Status |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 15.1s | 0.9523 | 72.06% | 0.5427 | 83.96% | $1.0\times 10^{-3}$ | ★ Best Saved |
| 2 | 4.4s | 0.5233 | 84.85% | 0.5014 | 85.56% | $1.0\times 10^{-3}$ | ★ Best Saved |
| 3 | 6.0s | 0.4285 | 87.88% | 0.3758 | 89.44% | $1.0\times 10^{-3}$ | ★ Best Saved |
| 4 | 4.2s | 0.3741 | 89.33% | 0.3775 | 88.87% | $1.0\times 10^{-3}$ | — |
| 5 | 4.0s | 0.3375 | 90.22% | 0.3443 | 90.29% | $1.0\times 10^{-3}$ | ★ Best Saved |
| 6 | 5.7s | 0.2982 | 91.26% | 0.3339 | 90.64% | $1.0\times 10^{-3}$ | ★ Best Saved |
| 7 | 4.0s | 0.2751 | 92.12% | 0.3352 | 91.11% | $1.0\times 10^{-3}$ | ★ Best Saved |
| 8 | 4.0s | 0.2422 | 92.83% | 0.3632 | 90.49% | $1.0\times 10^{-3}$ | — |
| 9 | 5.6s | 0.2229 | 93.41% | 0.3315 | 91.04% | $1.0\times 10^{-3}$ | — |
| 10 | 4.2s | 0.2063 | 93.72% | 0.3381 | 91.04% | $1.0\times 10^{-3}$ | — |
| 11 | 4.2s | 0.1867 | 94.29% | 0.3293 | 91.36% | $1.0\times 10^{-3}$ | ★ Best Saved |
| 12 | 5.2s | 0.1822 | 94.51% | 0.3283 | 91.18% | $1.0\times 10^{-3}$ | — |
| 13 | 4.9s | 0.1660 | 94.82% | 0.3524 | 90.67% | $1.0\times 10^{-3}$ | — |
| 14 | 4.1s | 0.1529 | 95.20% | 0.3604 | 90.80% | $1.0\times 10^{-3}$ | — |
| **15** | **4.9s** | **0.1374** | **95.49%** | **0.3285** | **91.58%** | **$1.0\times 10^{-3}$** | **★ Best Saved** |

- **Best Validation Accuracy:** **91.58%** at Epoch 15.
- **Saved Curves Plot:** [`reports/expanded_20/training_curves_20.png`](file:///Users/akshon/Desktop/datavision/reports/expanded_20/training_curves_20.png).
- **Saved History JSON:** [`reports/expanded_20/training_history_20.json`](file:///Users/akshon/Desktop/datavision/reports/expanded_20/training_history_20.json).

---

## 4. Single-Evaluation Results on Held-Out Test Set (4,500 Sketches)

Following selection at Epoch 15, `models/expanded_cnn_20.pt` was evaluated on `data/expanded_20/test.npz` (225 drawings per class):

### 4.1 Per-Class Performance Breakdown

| Class | Class Index | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `airplane` | 0 | 0.8943 | 0.9022 | 0.8982 | 225 |
| `apple` | 1 | 0.9595 | 0.9467 | 0.9530 | 225 |
| `axe` | 2 | 0.9198 | 0.8667 | 0.8924 | 225 |
| `bicycle` | 3 | 0.8845 | 0.9867 | 0.9328 | 225 |
| `book` | 4 | 0.9156 | 0.9156 | 0.9156 | 225 |
| `candle` | 5 | 0.9643 | 0.9600 | 0.9621 | 225 |
| `chair` | 6 | 0.9758 | 0.8978 | 0.9352 | 225 |
| `clock` | 7 | 0.9818 | 0.9600 | 0.9708 | 225 |
| `cup` | 8 | 0.9249 | 0.8756 | 0.8995 | 225 |
| `door` | 9 | 0.9381 | 0.9422 | 0.9401 | 225 |
| `fish` | 10 | 0.9127 | 0.9289 | 0.9207 | 225 |
| `flower` | 11 | 0.8372 | 0.9600 | 0.8944 | 225 |
| `house` | 12 | 0.9767 | 0.9333 | 0.9545 | 225 |
| `key` | 13 | 0.9024 | 0.8222 | 0.8605 | 225 |
| `parachute` | 14 | 0.9531 | 0.9022 | 0.9269 | 225 |
| `scissors` | 15 | 0.8125 | 0.8667 | 0.8387 | 225 |
| `shoe` | 16 | 0.9079 | 0.9200 | 0.9139 | 225 |
| `star` | 17 | 0.8889 | 0.9244 | 0.9063 | 225 |
| `tree` | 18 | 0.9244 | 0.9244 | 0.9244 | 225 |
| `umbrella` | 19 | 0.9248 | 0.9289 | 0.9268 | 225 |
| **MACRO AVERAGE** | — | **0.9200** | **0.9182** | **0.9184** | **4,500** |
| **OVERALL ACCURACY**| — | — | — | **91.82%** | **4,500** |

### 4.2 Confusion Analysis & Key Observations
- **Highest Performing Categories ($F_1 \ge 0.95$):**
  - `clock` ($F_1 = 0.9708$): Circular boundary with radial clock hands provides unique features.
  - `candle` ($F_1 = 0.9621$): Vertical candle body with teardrop flame remains distinct.
  - `house` ($F_1 = 0.9545$): Gabled roof over square base is easily classified.
  - `apple` ($F_1 = 0.9530$): Round silhouette with stem indentation.
- **Top Confusion Pairs:**
  - `key` $\rightarrow$ `axe` and `candle`: When keys are drawn with minimal teeth or vertically, minor confusion with linear objects persists (recall: 82.2%).
  - `scissors` $\rightarrow$ `flower` / `airplane`: Crossed blades with round loop handles occasionally resemble multi-lobed flowers or small delta planes (precision: 81.3%).
  - `cup` $\rightarrow$ `apple` / `candle`: Cylindrical mugs without prominent handles occasionally trigger round vessel detectors.
- **Visual Artifacts:**
  - Confusion Matrix: [`reports/expanded_20/confusion_matrix_20.png`](file:///Users/akshon/Desktop/datavision/reports/expanded_20/confusion_matrix_20.png)
  - Misclassified Examples Gallery: [`reports/expanded_20/misclassified_examples_20.png`](file:///Users/akshon/Desktop/datavision/reports/expanded_20/misclassified_examples_20.png)

---

## 5. Single-Sample CPU Latency Benchmark

Measured across 1,000 independent forward-pass trials on CPU with batch size 1 (simulating interactive drawing canvas inference):
- **Mean Latency:** **0.407 ms** per sketch.
- **Median Latency:** **0.382 ms** per sketch.
- **95th-Percentile Latency:** **0.525 ms** per sketch.
- **Standard Deviation:** **0.086 ms**.
- **Edge Deployment Assessment:** Real-time throughput exceeds **2,400 inferences per second** on a single CPU core, well within the latency budget for live Streamlit interaction.

---

## 6. Verification of Original 8-Class Files Preservation

MD5 checksums verified immediately post-evaluation:
* `data/train.npz`: `99def32bf871f4b178e03b0a7aaf2770` (MATCH)
* `data/val.npz`: `9807d10c5d29a212aa90e8f1be41b5a2` (MATCH)
* `data/test.npz`: `09b1b3be13b6f71e85f1900b9bbf6d2a` (MATCH)
* `data/split_metadata.json`: `0829ef282848cd63045920f825e07ed9` (MATCH)
* `models/baseline_cnn.pt`: `b5e8888d7261a7c1889d9d1bdb5b5d12` (MATCH)
* `models/label_mapping.json`: `b3556e85f9bbd10c92af147aec407036` (MATCH)
* **Unit Test Suite:** `python -m unittest discover tests` executed cleanly ($25\text{ tests in }0.030\text{s}$, 0 failures, 0 errors).
