# STAGE F GAMEPLAY REPORT: 3-DIFFICULTY RIDDLE GAME
**Project:** Draw to Survive: An AI-Powered Drawing Adventure  
**Course:** Edge AI and Robotics: Data Center Vision  
**Date:** October 14, 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

Stage F transformed the 20-room linear story adventure into a drawing-based riddle challenge with three distinct difficulty tiers:
- **Easy:** 5 riddles
- **Medium:** 5 riddles
- **Hard:** 5 riddles  
- **Total:** 15 playable riddles

Each difficulty presents the player with a compact 5-riddle trial where answers are discovered through poetic metaphors and submitted via the real-time AI sketch-recognition canvas.

The transformation was executed strictly within gameplay and challenge logic:
- **Zero Retraining or Model Modification:** Checkpoints `models/expanded_cnn_20.pt` and `models/baseline_cnn.pt` remain untouched.
- **Zero Dataset Alteration:** Raw QuickDraw datasets and NPZ files remain unmodified.
- **Backward Compatibility:** Original 8-class facility mode (`CHALLENGES_8`) remains fully operational and selectable.
- **100% Test Suite Verification:** 56 unit tests (25 original + 16 Stage D + 15 Stage F) execute and pass cleanly in **0.093s**.

---

## 2. Files Changed

| File Path | Description of Changes |
|:---|:---|
| [`src/challenges.py`](file:///Users/akshon/Desktop/datavision/src/challenges.py) | Added optional `riddle` and `intended_answer` fields to `Challenge` dataclass. Defined `RIDDLES_EASY`, `RIDDLES_MEDIUM`, `RIDDLES_HARD`, `RIDDLES_15`, and `RIDDLES_BY_DIFFICULTY`. Added `get_riddles(difficulty)` and updated `get_challenges(version, difficulty)` and `validate_challenges(version)` to strictly validate all 15 riddle solutions against the 20-class mapping. |
| [`src/game_engine.py`](file:///Users/akshon/Desktop/datavision/src/game_engine.py) | Added optional `difficulty` parameter to `GameEngine.__init__`, stored `self.difficulty`, and exposed `"difficulty"` in `get_summary()`. |
| [`app.py`](file:///Users/akshon/Desktop/datavision/app.py) | Integrated difficulty state (`st.session_state.difficulty`), sidebar difficulty selector, Welcome screen difficulty tier selection cards, riddle HUD metrics (`Riddle 1 / 5`), styled riddle quote cards, dynamic success/victory/game-over screens, and restart/difficulty switching controls. |
| [`tests/test_stage_f_riddles.py`](file:///Users/akshon/Desktop/datavision/tests/test_stage_f_riddles.py) | New test suite with 15 automated test cases verifying riddle counts, answer mappings, API helpers, difficulty progression, scoring idempotency, blank canvas safety, threshold rejection, game over, restart, and model compatibility. |

---

## 3. Riddle Directory by Difficulty

All 15 riddles are structured as `Challenge` objects containing stable unique IDs, titles, poetic riddle texts, single verified answers, failure hints, difficulty identifiers, and 3-tier progressive hints.

### 3.1 Easy Mode (5 Riddles)

| # | ID | Title | Riddle Text | Intended Answer | Accepted Label |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | `riddle_easy_01_candle` | The Shrinking Storyteller | *"I speak without a voice, and the more I tell, the less of me remains. I can brighten a room, but a little breath can silence me. What am I?"* | `candle` | `["candle"]` |
| 2 | `riddle_easy_02_umbrella` | The Portable Roof | *"I open up when the sky breaks down. I have ribs but no bones, and I protect you without ever stepping outside myself. What am I?"* | `umbrella` | `["umbrella"]` |
| 3 | `riddle_easy_03_fish` | The Silent Swimmer | *"I travel without feet, breathe without lungs, and wear my own shimmering coat. Take me out of my world, and my journey ends. What am I?"* | `fish` | `["fish"]` |
| 4 | `riddle_easy_04_bicycle` | The Balanced Traveller | *"I have two circles but no face, a chain but no prisoner, and I can carry you far without ever taking a step. What am I?"* | `bicycle` | `["bicycle"]` |
| 5 | `riddle_easy_05_book` | The Quiet World | *"I can take you to places you have never visited and introduce you to people who never existed. I stay perfectly still while your mind travels. What am I?"* | `book` | `["book"]` |

### 3.2 Medium Mode (5 Riddles)

| # | ID | Title | Riddle Text | Intended Answer | Accepted Label |
|:---:|:---|:---|:---|:---:|:---:|
| 6 | `riddle_med_01_clock` | The Relentless Circle | *"I revisit the same places every day, yet never return to yesterday. Everyone watches me move, but no one can ask me to wait. What am I?"* | `clock` | `["clock"]` |
| 7 | `riddle_med_02_key` | The Tiny Negotiator | *"I have teeth but cannot bite. I can end a stranger's visit or begin a friend's, yet I never decide who is welcome. What am I?"* | `key` | `["key"]` |
| 8 | `riddle_med_03_flower` | The Colourful Clock | *"I can mark a season without numbers, attract visitors without a sign, and offer beauty before I ever offer fruit. What am I?"* | `flower` | `["flower"]` |
| 9 | `riddle_med_04_shoe` | The Loyal Follower | *"I follow you everywhere but never choose the route. I touch the ground so you don't have to, and I'm happiest when I fit you perfectly. What am I?"* | `shoe` | `["shoe"]` |
| 10 | `riddle_med_05_house` | The Keeper of Secrets | *"I have a face that strangers see, rooms that guests may enter, and a door that can keep the whole world out. What am I?"* | `house` | `["house"]` |

### 3.3 Hard Mode (5 Riddles)

| # | ID | Title | Riddle Text | Intended Answer | Accepted Label |
|:---:|:---|:---|:---|:---:|:---:|
| 11 | `riddle_hard_01_scissors` | The Double-Edged Decision | *"I need two partners to do my job, but they spend their time moving against each other. Bring them together and something whole becomes two. What am I?"* | `scissors` | `["scissors"]` |
| 12 | `riddle_hard_02_parachute` | The Art of Falling Slowly | *"My purpose is to make a fall last longer. I catch the air, turn speed into patience, and help a traveller meet the ground gently. What am I?"* | `parachute` | `["parachute"]` |
| 13 | `riddle_hard_03_airplane` | The Sky's Contradiction | *"I am heavier than the air I cross, yet I can stay above it. I carry people over oceans without swimming, and my wings are not for flapping. What am I?"* | `airplane` | `["airplane"]` |
| 14 | `riddle_hard_04_tree` | The Upside-Down Kingdom | *"My feet are hidden, my arms reach for the sky, and every year I may dress in a different way. Birds can call me home, though I never build a house. What am I?"* | `tree` | `["tree"]` |
| 15 | `riddle_hard_05_axe` | The Woodcutter's Paradox | *"I make my mark by separating what nature joined. The harder I meet a tree, the smaller its trunk becomes, while my own shape stays almost the same. What am I?"* | `axe` | `["axe"]` |

---

## 4. Answer-to-Label Validation

Every intended riddle solution was audited against the model's vocabulary (`models/label_mapping_20.json`):

| Answer | Model Class Index | Difficulty Tier | Validation Status |
|:---|:---:|:---:|:---:|
| `airplane` | 0 | Hard | Verified Match |
| `axe` | 2 | Hard | Verified Match |
| `bicycle` | 3 | Easy | Verified Match |
| `book` | 4 | Easy | Verified Match |
| `candle` | 5 | Easy | Verified Match |
| `clock` | 7 | Medium | Verified Match |
| `fish` | 10 | Easy | Verified Match |
| `flower` | 11 | Medium | Verified Match |
| `house` | 12 | Medium | Verified Match |
| `key` | 13 | Medium | Verified Match |
| `parachute` | 14 | Hard | Verified Match |
| `scissors` | 15 | Hard | Verified Match |
| `shoe` | 16 | Medium | Verified Match |
| `tree` | 18 | Hard | Verified Match |
| `umbrella` | 19 | Easy | Verified Match |

- **Exact 1:1 Mapping:** Each riddle accepts strictly its single genuine answer. No artificial multiple aliases or unmapped labels exist.
- **Import Validation:** `validate_challenges(version=20)` executes automatically on module load; any mismatch will raise a `ValueError` immediately.

---

## 5. Gameplay Rules & Scoring Architecture

1. **Difficulty Selection:** The player selects Easy, Medium, or Hard on the Welcome screen or via the sidebar. Exactly 5 riddles are loaded for that trial.
2. **Display & Submission:** One riddle is shown at a time with a centered canvas pad. The player sketches their solution and clicks **Submit Drawing**.
3. **Inference Pipeline:** The drawing is preprocessed to $28 \times 28$ grayscale and passed through `QuickDrawPredictor(model_version=20)`. The predicted class and softmax confidence are evaluated by `GameEngine.submit_attempt`.
4. **Scoring Rules:**
   - `+100 points` base for every solved riddle.
   - `+50 points` first-attempt bonus if solved on the initial submission.
   - `+25 points` per remaining life awarded upon trial completion (`VICTORY`).
   - Maximum possible score for a flawless 5-riddle run: $5 \times 150 + 3 \times 25 = 825 \text{ points}$.
5. **Life Rules:**
   - The player starts with 3 lives.
   - An incorrect prediction or a prediction below the confidence cutoff (default 40%) costs 1 life.
   - A completely blank canvas is safe and does **not** consume a life.
   - Reaching 0 lives transitions to `GAME_OVER`.
6. **Progression & Completion:**
   - Solving a riddle transitions to `ROOM_SUCCESS` with an advance button (`➡️ Continue to Next Riddle`).
   - Solving Riddle 5 transitions to `VICTORY` with a completion scorecard and celebration balloons.
7. **Rerun & Session Stability:**
   - Submissions cannot duplicate points: solved room IDs are tracked in a set (`solved_room_ids`), preventing score inflation on page reloads.
   - Canvas state is isolated per riddle with an incremental `canvas_nonce`.
   - Dedicated buttons allow restarting the current difficulty or switching to a different tier.

---

## 6. Automated Test Suite Results

Ran `.venv/bin/python -m unittest discover tests`:

```
[Test Canvas Prediction] Predicted: key (99.19%)
[Top-3 Predictions]: [('key', 0.9918698072433472), ('axe', 0.0038272037636488676), ('candle', 0.001571754808537662)]
........................................................
----------------------------------------------------------------------
Ran 56 tests in 0.093s

OK
```

### 6.1 Test Suite Breakdown
| Test File | Test Count | Pass Rate | Scope |
|:---|:---:|:---:|:---|
| `test_game.py` | 13 | 100% | Legacy 8-class engine mechanics & scoring rules |
| `test_game_state_exhaustive.py` | 8 | 100% | State transitions, boundary edge cases, restart semantics |
| `test_playthrough.py` | 2 | 100% | Multi-room progression, game over, victory scoring |
| `test_canvas_integration.py` | 2 | 100% | Empty canvas safety, key drawing recognition & point award |
| `test_expanded_20.py` | 16 | 100% | 20-class model loading, 20 logits, 20 facility challenges |
| `test_stage_f_riddles.py` **(NEW)** | 15 | 100% | Riddle counts, answer-label matching, 3-difficulty progression, life safety, scoring idempotency, restart/switch |
| **Total** | **56** | **100%** | **Zero failures, zero regressions across all modes** |

### 6.2 Simulated Model Playthrough Verification
Simulated automated playthroughs of all 3 difficulties using real test set samples from `data/expanded_20/test.npz` and `QuickDrawCNN`:
- **Easy Mode:** 5/5 riddles solved $\rightarrow$ `VICTORY` (Score: 825 pts, Lives: 3/3).
- **Medium Mode:** 5/5 riddles solved $\rightarrow$ `VICTORY` (Score: 825 pts, Lives: 3/3).
- **Hard Mode:** 5/5 riddles solved $\rightarrow$ `VICTORY` (Score: 825 pts, Lives: 3/3).

---

## 7. Safety Invariance Audit

Cryptographic MD5 verification of baseline models and datasets:
- `models/baseline_cnn.pt`: `b5e8888d7261a7c1889d9d1bdb5b5d12` (Unchanged)
- `models/label_mapping.json`: `b3556e85f9bbd10c92af147aec407036` (Unchanged)
- `models/expanded_cnn_20.pt`: `e98fde5bb10367ae0a03fab09e0ffa6c` (Unchanged)
- `models/label_mapping_20.json`: `2650c8c9b3bd3d4b09a374fed06b8d85` (Unchanged)
- `data/{train,val,test}.npz`: Exact hash match (Unchanged)
- `data/expanded_20/{train,val,test}.npz`: Exact hash match (Unchanged)

---

## 8. Known Limitations & Unverified Behaviour

1. **Browser Testing Scope:** Verification was conducted via programmatic Streamlit execution tests and Python unit test suites. In accordance with safety instructions, live interactive human browser playtesting has not yet been conducted and is reserved for the UI polish stage.
2. **Ambiguity on Abstract Metaphors:** In Hard mode, riddles such as *The Upside-Down Kingdom* (`tree`) or *The Art of Falling Slowly* (`parachute`) require the player to deduce the object from abstract paradoxes. Progressive hints provide specific structural drawing advice if the player struggles.

---

## 9. Conclusion

Stage F gameplay and challenge logic implementation is complete and verified. The riddle system is fully playable across Easy, Medium, and Hard tiers, backed by 56 passing unit tests and zero model regressions. Ready for user inspection.
