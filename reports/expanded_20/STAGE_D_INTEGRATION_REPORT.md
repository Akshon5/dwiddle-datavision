# STAGE D INTEGRATION REPORT: 20-CLASS MODEL & 20-ROOM ADVENTURE
**Project:** Draw to Survive: An AI-Powered Drawing Adventure  
**Course:** Edge AI and Robotics: Data Center Vision  
**Date:** October 14, 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

In Stage D, the trained 20-category Convolutional Neural Network checkpoint (`models/expanded_cnn_20.pt`) was integrated into the *Draw to Survive* game ecosystem without modifying model weights, breaking backward compatibility, or compromising historical data. 

The application now seamlessly supports both the **Expanded 20-Class Adventure (20 Rooms)** and the **Original 8-Class Facility (8 Rooms)** via a dynamic runtime selector.

### Key Milestones Completed:
1. **Model & Mapping Strict Pairing:** Connected `models/expanded_cnn_20.pt` with `models/label_mapping_20.json` (and `data/expanded_20/label_mapping.json`), verifying 20 output logits and contiguous $[0, 19]$ label mapping.
2. **20 Coherent Challenge Sectors:** Expanded challenges in `src/challenges.py` from 8 to 20 logically consistent, progressively challenging escape-room scenarios. Fixed previous logical anomalies (e.g., room 4 ventilation conduit replaced with underground transit tunnel).
3. **Progressive Hint System:** Added 3-tier progressive hints to all rooms to provide scaffolded player assistance without breaking immersion.
4. **Interactive UI Upgrades (`app.py`):** Added a model version selector dropdown in the sidebar, 20-class visual icon indicators, sector-specific tactical drawing guidance, and dynamic victory scoring scaling to 20 rooms.
5. **100% Test Suite Verification:** Developed 16 new automated regression tests in `tests/test_expanded_20.py`. Ran all 41 test cases (25 legacy + 16 new) with 100% pass rate in **0.098 seconds**.
6. **Full Playthrough Simulation:** Verified end-to-end 20-room progression from Room 1 to Room 20 using real model inference, ending in `VICTORY` with all 3 lives intact and 3,075 points.
7. **Safety Invariance:** Verified all original baseline files (`models/baseline_cnn.pt`, `models/label_mapping.json`, and original datasets) remain 100% bit-for-bit identical to their original MD5 hashes.

---

## 2. Checkpoint & Architecture Verification

### 2.1 Model Specifications
- **Checkpoint Location:** `models/expanded_cnn_20.pt` (Size: 981 KB)
- **Architecture:** `QuickDrawCNN` with 3 Convolutional Blocks + Flatten + FC Head
- **Input Dimension:** $1 \times 28 \times 28$ grayscale float32 normalized to $[0, 1]$
- **Output Dimension:** $1 \times 20$ unnormalized logits
- **Parameters:** 243,060 trainable parameters

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 28, 28]             288
       BatchNorm2d-2           [-1, 32, 28, 28]              64
              ReLU-3           [-1, 32, 28, 28]               0
         MaxPool2d-4           [-1, 32, 14, 14]               0
            Conv2d-5           [-1, 64, 14, 14]          18,432
       BatchNorm2d-6           [-1, 64, 14, 14]             128
              ReLU-7           [-1, 64, 14, 14]               0
         MaxPool2d-8             [-1, 64, 7, 7]               0
            Conv2d-9            [-1, 128, 7, 7]          73,728
      BatchNorm2d-10            [-1, 128, 7, 7]             256
             ReLU-11            [-1, 128, 7, 7]               0
        MaxPool2d-12            [-1, 128, 3, 3]               0
          Flatten-13                 [-1, 1152]               0
          Dropout-14                 [-1, 1152]               0
           Linear-15                  [-1, 128]         147,584
             ReLU-16                  [-1, 128]               0
          Dropout-17                  [-1, 128]               0
           Linear-18                   [-1, 20]           2,580
================================================================
Total params: 243,060
```

### 2.2 Class Label Mapping (20 Classes)
Verified exact match between `data/expanded_20/label_mapping.json` and `models/label_mapping_20.json`:

| Index | Class Name | Category Group | Origin |
|:---:|:---|:---|:---:|
| 0 | `airplane` | Vehicles & Transport | Original 8 |
| 1 | `apple` | Food & Biology | Original 8 |
| 2 | `axe` | Tools & Hardware | Original 8 |
| 3 | `bicycle` | Vehicles & Transport | Original 8 |
| 4 | `book` | Household & Utility | New 12 |
| 5 | `candle` | Household & Utility | Original 8 |
| 6 | `chair` | Furniture & Structures | New 12 |
| 7 | `clock` | Hardware & Electronics | New 12 |
| 8 | `cup` | Food & Containers | New 12 |
| 9 | `door` | Furniture & Structures | Original 8 |
| 10 | `fish` | Animals & Nature | New 12 |
| 11 | `flower` | Nature & Plants | New 12 |
| 12 | `house` | Buildings & Architecture | New 12 |
| 13 | `key` | Tools & Hardware | Original 8 |
| 14 | `parachute` | Safety & Airborne | Original 8 |
| 15 | `scissors` | Tools & Hardware | New 12 |
| 16 | `shoe` | Apparel & Gear | New 12 (Approved replacement for hammer) |
| 17 | `star` | Symbols & Shapes | New 12 |
| 18 | `tree` | Nature & Plants | New 12 |
| 19 | `umbrella` | Gear & Utility | New 12 |

---

## 3. Challenge Design & Narrative Expansion

All 20 rooms were designed to form a single continuous narrative arc: escaping from deep subterranean containment vaults up through laboratory facilities, maintenance tunnels, and out across an abandoned mountain airfield.

### 3.1 Complete 20-Sector Directory

| Sector # | ID | Sector Title | Accepted Solutions | Difficulty | Objective Summary |
|:---:|:---|:---|:---|:---:|:---|
| **01** | `sector_01_cell` | The Armored Cell | `key`, `door` | Introductory | Unlock brass deadbolt or breach containment. |
| **02** | `sector_02_timberway` | The Collapsed Timberway | `axe` | Introductory | Chop heavy structural beams blocking corridor. |
| **03** | `sector_03_blackout` | Pitch-Black Basement | `candle` | Introductory | Provide flame illumination to navigate high-voltage floor. |
| **04** | `sector_04_transit` | Subterranean Transit Tunnel | `bicycle` | Moderate | Pedal rail trackway to outrun automated bulkhead closure. |
| **05** | `sector_05_quarantine` | Medical Quarantine Airlock | `apple` | Moderate | Satisfy organic nutrient sensor to clear bio-quarantine. |
| **06** | `sector_06_archives` | Encrypted Archives Terminal | `book` | Moderate | Provide cipher manual reference to override security door. |
| **07** | `sector_07_breaker` | High-Voltage Breaker Room | `chair` | Moderate | Stand on elevated platform to reach overhead emergency lever. |
| **08** | `sector_08_reactor` | Overheating Reactor Core | `clock` | Moderate | Calibrate countdown timer to trigger pressure relief valve. |
| **09** | `sector_09_tap` | Decontamination Dispenser | `cup` | Moderate | Collect neutralizer solvent to dissolve caustic lock residue. |
| **10** | `sector_10_reservoir` | Submerged Drainage Aquifer | `fish` | Moderate | Deploy biomimetic reconnaissance to find underwater sluice. |
| **11** | `sector_11_hydroponics` | Toxic Botany Lab | `flower` | Moderate | Extract botanical reagent to neutralize airborne spore toxin. |
| **12** | `sector_12_blastgate` | Reinforced Security Airlock | `door`, `key` | Moderate | Disengage heavy vault seal or bypass locking latch. |
| **13** | `sector_13_tripwire` | Explosive Demolition Grid | `scissors` | Challenging | Snip emergency detonation tripwire to defuse charges. |
| **14** | `sector_14_floorgrid` | Electrified Floor Plates | `shoe` | Challenging | Equip rubber-soled footwear to safely cross electrified floor. |
| **15** | `sector_15_stellar` | Astrological Iris Vault | `star` | Challenging | Align 5-pointed celestial emblem into mechanical gear lock. |
| **16** | `sector_16_courtyard` | Courtyard Antenna Mast | `tree`, `flower` | Challenging | Disguise emergency radio antenna as courtyard vegetation. |
| **17** | `sector_17_acid` | Acid Deluge Catwalk | `umbrella` | Challenging | Deflect corrosive chemical droplets overhead. |
| **18** | `sector_18_radar` | Radar Telemetry Array | `house` | Challenging | Lock beacon onto abandoned surface refuge coordinate. |
| **19** | `sector_19_precipice` | Mountain Ventilation Shaft | `parachute` | Challenging | Leap across yawning chasm to reach mountain landing. |
| **20** | `sector_20_airfield` | The Mountain Runway | `airplane` | Challenging | Power up twin turboprops and evacuate the facility! |

### 3.2 Progressive Hint Architecture
Every challenge now embeds a 3-tier progressive hint list:
- **Tier 1 (Subtle Context):** Atmospheric observation pointing to functional need.
- **Tier 2 (Functional Category):** Specific hint about object type or purpose.
- **Tier 3 (Explicit Drawing Tip):** Visual geometric breakdown (e.g., *"Draw two circular wheels connected by a triangular frame"*).

---

## 4. UI Implementation Details (`app.py`)

1. **Model Version Selector:**
   Placed at the top of the sidebar. Allows the user to toggle between:
   - `20 Classes (Expanded Adventure)` $\rightarrow$ loads `expanded_cnn_20.pt` and 20 sectors.
   - `8 Classes (Original Facility)` $\rightarrow$ loads `baseline_cnn.pt` and 8 legacy sectors.
2. **Visual Class Badges:**
   Configured 20 distinctive emoji icons matching all categories (`✈️ Airplane`, `🍎 Apple`, `🪓 Axe`, `🚲 Bicycle`, `📖 Book`, `🕯️ Candle`, `🪑 Chair`, `⏰ Clock`, `☕ Cup`, `🚪 Door`, `🐟 Fish`, `🌸 Flower`, `🏠 House`, `🔑 Key`, `🪂 Parachute`, `✂️ Scissors`, `👟 Shoe`, `⭐ Star`, `🌲 Tree`, `☂️ Umbrella`).
3. **Sector Clues Expander:**
   Added `🕵️ Sector Intel & Clues (Progressive Hints)` accordion below each scenario briefing.
4. **Contextual Drawing Guidance:**
   Replaced static 6-item advice with dynamic tips generated specifically for the current sector's accepted solutions.
5. **Dynamic Scoring & Victory Screen:**
   Replaced hardcoded `8` with `summary['total_rooms']` throughout HUD, scorecard metrics, and celebration banners.

---

## 5. Test Suite & Verification Results

### 5.1 Automated Test Suite Execution
Ran `.venv/bin/python -m unittest discover tests`:

```
[Test Canvas Prediction] Predicted: key (99.19%)
[Top-3 Predictions]: [('key', 0.9918698072433472), ('axe', 0.0038272037636488676), ('candle', 0.001571754808537662)]
.........................................
----------------------------------------------------------------------
Ran 41 tests in 0.098s

OK
```

### 5.2 Test Suite Breakdown
| Test Module | Test Count | Pass Rate | Key Verification Focus |
|:---|:---:|:---:|:---|
| `test_game.py` | 13 | 100% | Legacy 8-class game engine, scoring rules, life deductions |
| `test_game_state_exhaustive.py` | 8 | 100% | State transitions, boundary edge cases, restart semantics |
| `test_playthrough.py` | 2 | 100% | Multi-room progression, game over, victory scoring |
| `test_canvas_integration.py` | 2 | 100% | Empty canvas safety, key drawing recognition & point award |
| `test_expanded_20.py` **(NEW)** | 16 | 100% | 20-class loading, 20 logits, label mapping, 20 challenges, scoring, hints, full victory |
| **Total** | **41** | **100%** | **Zero failures, zero regressions across both models** |

### 5.3 Simulated 20-Room Playthrough Verification
Tested full end-to-end 20-room progression using real test dataset samples and the 20-class CNN:

```
Starting full 20-room verification playthrough. Total rooms: 20
  Room 01 (sector_01_cell): SOLVED | Target: key -> Pred: key (45.2%) (+50 1st Try Bonus!)
  Room 02 (sector_02_timberway): SOLVED | Target: axe -> Pred: axe (99.0%) (+50 1st Try Bonus!)
  Room 03 (sector_03_blackout): SOLVED | Target: candle -> Pred: candle (100.0%) (+50 1st Try Bonus!)
  Room 04 (sector_04_transit): SOLVED | Target: bicycle -> Pred: bicycle (100.0%) (+50 1st Try Bonus!)
  Room 05 (sector_05_quarantine): SOLVED | Target: apple -> Pred: apple (100.0%) (+50 1st Try Bonus!)
  Room 06 (sector_06_archives): SOLVED | Target: book -> Pred: book (100.0%) (+50 1st Try Bonus!)
  Room 07 (sector_07_breaker): SOLVED | Target: chair -> Pred: chair (100.0%) (+50 1st Try Bonus!)
  Room 08 (sector_08_reactor): SOLVED | Target: clock -> Pred: clock (100.0%) (+50 1st Try Bonus!)
  Room 09 (sector_09_tap): SOLVED | Target: cup -> Pred: cup (100.0%) (+50 1st Try Bonus!)
  Room 10 (sector_10_reservoir): SOLVED | Target: fish -> Pred: fish (99.7%) (+50 1st Try Bonus!)
  Room 11 (sector_11_hydroponics): SOLVED | Target: flower -> Pred: flower (99.8%) (+50 1st Try Bonus!)
  Room 12 (sector_12_blastgate): SOLVED | Target: door -> Pred: door (100.0%) (+50 1st Try Bonus!)
  Room 13 (sector_13_tripwire): SOLVED | Target: scissors -> Pred: scissors (99.8%) (+50 1st Try Bonus!)
  Room 14 (sector_14_floorgrid): SOLVED | Target: shoe -> Pred: shoe (70.1%) (+50 1st Try Bonus!)
  Room 15 (sector_15_stellar): SOLVED | Target: star -> Pred: star (100.0%) (+50 1st Try Bonus!)
  Room 16 (sector_16_courtyard): SOLVED | Target: tree -> Pred: tree (83.8%) (+50 1st Try Bonus!)
  Room 17 (sector_17_acid): SOLVED | Target: umbrella -> Pred: umbrella (100.0%) (+50 1st Try Bonus!)
  Room 18 (sector_18_radar): SOLVED | Target: house -> Pred: house (100.0%) (+50 1st Try Bonus!)
  Room 19 (sector_19_precipice): SOLVED | Target: parachute -> Pred: parachute (79.1%) (+50 1st Try Bonus!)
  Room 20 (sector_20_airfield): SOLVED | Target: airplane -> Pred: airplane (100.0%) (+50 1st Try Bonus!)
============================================================
Final Playthrough Result:
  Game State: VICTORY
  Rooms Solved: 20/20
  First-Try Bonuses: 20
  Lives Remaining: 3/3
  Total Final Score: 3075 pts
============================================================
```

---

## 6. Safety Compliance & Hash Verification

To guarantee adherence to the project's non-negotiable safety rules, cryptographic MD5 hashes of all original project artifacts were audited:

| File Path | Recorded Baseline MD5 | Post-Stage D MD5 | Status |
|:---|:---:|:---:|:---:|
| `models/baseline_cnn.pt` | `b5e8888d7261a7c1889d9d1bdb5b5d12` | `b5e8888d7261a7c1889d9d1bdb5b5d12` | **100% UNCHANGED** |
| `models/label_mapping.json` | `b3556e85f9bbd10c92af147aec407036` | `b3556e85f9bbd10c92af147aec407036` | **100% UNCHANGED** |
| `data/train.npz` | `99def32bf871f4b178e03b0a7aaf2770` | `99def32bf871f4b178e03b0a7aaf2770` | **100% UNCHANGED** |
| `data/val.npz` | `9807d10c5d29a212aa90e8f1be41b5a2` | `9807d10c5d29a212aa90e8f1be41b5a2` | **100% UNCHANGED** |
| `data/test.npz` | `09b1b3be13b6f71e85f1900b9bbf6d2a` | `09b1b3be13b6f71e85f1900b9bbf6d2a` | **100% UNCHANGED** |

Zero model retraining occurred during Stage D. The system cleanly preserves historical reports, baseline datasets, and allows instantaneous toggling between the 8-class and 20-class game modes.

---

## 7. Conclusion & Readiness

Stage D is complete and fully verified. The application is playable across all 20 categories, backed by 41 passing unit tests and live verified Streamlit execution. Awaiting user review before proceeding to Stage E (Full Game Testing, Debugging & Submission Polish).
