# STAGE 5: COMPREHENSIVE TEST PLAN & AUDIT SPECIFICATION

**Project:** Draw to Survive: An AI-Powered Drawing Adventure  
**Course:** Edge AI and Robotics: Data Center Vision  
**Audit Date:** October 13, 2026  
**Objective:** End-to-end verification, edge-case testing, drawing recognition evaluation, and polish before college lab submission.

---

## 1. Scope & Preservation Invariants

- **Preserved Artifacts:**
  - Neural network weights: `models/baseline_cnn.pt` (no retraining or architecture modification)
  - Dataset partitions: `data/train.npz`, `data/val.npz`, `data/test.npz` (held-out test set untouched)
  - Historical reports: `STAGE1_DATASET_REPORT.md`, Stage 2 training history, `STAGE3_CANVAS_VALIDATION.md`
- **Scope:**
  - Automated game-state transitions across all 8 crisis sectors.
  - Boundary condition validation (last life, final room, repeated button inputs, empty canvas submissions).
  - Fresh real-sketch drawing recognition benchmark (6 diverse sketches per class = 48 total).
  - Preprocessing pipeline audit (polarity detection, bounding-box cropping, bilinear resizing to 28×28, normalization).
  - Usability, layout, and visual feedback polish in `app.py`.

---

## 2. Full Game-State Test Matrix

| ID | Test Scenario | Module / Component | Expected Behavior | Actual Result |
| :---: | :--- | :--- | :--- | :---: |
| **TC-01** | Initial Welcome State | `GameEngine.__init__` | Engine initializes with `lives=3`, `score=0`, `current_room_idx=0`, state=`"WELCOME"`. | Pending |
| **TC-02** | Room 1: Armored Cell | `GameEngine.submit_attempt` | Accepts `["key", "door"]`. Solving awards 150 pts (100 base + 50 first-try). | Pending |
| **TC-03** | Room 2: Collapsed Timberway | `GameEngine.submit_attempt` | Accepts `["axe"]`. Solving awards points. | Pending |
| **TC-04** | Room 3: Pitch-Black Basement | `GameEngine.submit_attempt` | Accepts `["candle"]`. Solving awards points. | Pending |
| **TC-05** | Room 4: Ventilation Conduit | `GameEngine.submit_attempt` | Accepts `["bicycle"]`. Solving awards points. | Pending |
| **TC-06** | Room 5: Medical Quarantine | `GameEngine.submit_attempt` | Accepts `["apple"]`. Solving awards points. | Pending |
| **TC-07** | Room 6: Security Checkpoint | `GameEngine.submit_attempt` | Accepts `["key", "door"]`. Both alternatives solve cleanly. | Pending |
| **TC-08** | Room 7: Rooftop Chasm | `GameEngine.submit_attempt` | Accepts `["parachute"]`. Solving awards points. | Pending |
| **TC-09** | Room 8: Final Runway | `GameEngine.submit_attempt` | Accepts `["airplane"]`. Advancing room transitions to `"VICTORY"`. | Pending |
| **TC-10** | Incorrect Class Deduction | `GameEngine.submit_attempt` | Submitting wrong class reduces lives by 1, increments `attempts_this_room`, preserves score. | Pending |
| **TC-11** | Low Confidence Rejection | `GameEngine.submit_attempt` | Non-empty drawing with confidence $< \text{threshold}$ consumes 1 life, returns hint. | Pending |
| **TC-12** | Blank Canvas Safety | `GameEngine.submit_attempt` | Submitting blank canvas consumes 0 lives, does not advance attempt counter. | Pending |
| **TC-13** | First-Attempt Bonus Rule | `GameEngine.submit_attempt` | +50 pts awarded on attempt 0. Submitting wrong first, then correct awards +100 pts only. | Pending |
| **TC-14** | Duplicate Scoring Guard | `GameEngine.submit_attempt` | Repeated submission in `ROOM_SUCCESS` cannot award points twice. | Pending |
| **TC-15** | Boundary: Final Life (1 -> 0) | `GameEngine.submit_attempt` | Failing with 1 life transitions immediately to `"GAME_OVER"`. Bounded $\ge 0$. | Pending |
| **TC-16** | Victory Lives Bonus | `GameEngine.advance_room` | On Room 8 completion, adds $25 \times \text{lives}$ bonus to final score. | Pending |
| **TC-17** | State Reset on Restart | `GameEngine.restart_game` | Completely resets score, lives, current room, solved rooms, and state to `"WELCOME"`. | Pending |
| **TC-18** | Challenge Class Validity | `src/challenges.py` | 100% of accepted classes exist in `models/label_mapping.json`. | Pending |

---

## 3. Real Drawing-Recognition Evaluation Protocol

We will evaluate **48 fresh, newly constructed sketches** (6 per class) on a simulated $300 \times 300$ drawing canvas:
- **Classes (8):** `airplane`, `apple`, `axe`, `bicycle`, `candle`, `door`, `key`, `parachute`.
- **Systematic Variations:**
  1. *Standard/Canonical* (balanced stroke width, centered)
  2. *Thick stroke* (12–16 px brush, bold lines)
  3. *Thin stroke* (3–5 px fine lines)
  4. *Rotated/Angled* ($\pm 25^\circ$ tilt)
  5. *Off-center/Scale variation* (smaller or translated off-center)
  6. *Stylistic alternative* (e.g., bitten apple, double-sided axe, panelled door, paraglider canopy)
- **Metrics Collected:**
  - True Class, Predicted Class, Confidence (%)
  - Correctness ($\text{top-1} == \text{true}$)
  - Confidence threshold status ($\ge 40\%$ accepted vs $< 40\%$ rejected)
  - Confusion patterns and visual inspectability.

---

## 4. Preprocessing & Confidence Audit

1. **Polarity Check:** Verify whether corner sampling accurately detects white canvas vs dark canvas.
2. **Bounding Box & Padding:** Verify that extreme aspect ratio sketches (e.g. long thin candle or wide airplane) are centered into squares with padding without clipping.
3. **Resampling Quality:** Verify bilinear interpolation downsamples smoothly to $28 \times 28$ uint8.
4. **Softmax Behavior:** Document whether 40% threshold effectively filters ambiguous sketches or if another cutoff is warranted.

---

## 5. UI & Usability Audit Criteria

1. **Visual Contrast:** Verify readability of text on dark backgrounds.
2. **Canvas Reset:** Confirm that advancing rooms increments `canvas_nonce` to prevent stale strokes from persisting.
3. **Disabled States / Button Guards:** Confirm that users cannot trigger invalid state transitions.
4. **HUD Visibility:** Lives, Score, Room number, and Mission Prompt clearly legible on standard displays.

---
*Execution results will be updated into this test plan and summarized in `reports/STAGE5_FINAL_VALIDATION.md`.*
