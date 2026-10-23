"""
Unit and integration tests for Stage F: 3-Difficulty Riddle Game.
Tests riddle definitions (Easy: 5, Medium: 5, Hard: 5 = 15 total),
difficulty selection, answer-to-model validation, GameEngine riddle progression,
scoring idempotency, lives management, restart/switching, and 8-class compatibility.
"""

import json
import os
import unittest
import numpy as np
import torch

from src.challenges import (
    RIDDLES_EASY,
    RIDDLES_MEDIUM,
    RIDDLES_HARD,
    RIDDLES_15,
    RIDDLES_BY_DIFFICULTY,
    CHALLENGES_8,
    CHALLENGES_20,
    get_riddles,
    get_challenges,
    get_challenge,
    validate_challenges
)
from src.game_engine import (
    GameEngine,
    POINTS_PER_ROOM,
    FIRST_TRY_BONUS,
    BONUS_PER_LIFE
)
from src.inference import QuickDrawPredictor


class TestStageFRiddles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.label_path_20 = os.path.join("models", "label_mapping_20.json")
        with open(cls.label_path_20, "r") as f:
            cls.label_mapping_20 = json.load(f)
        cls.predictor = QuickDrawPredictor(model_version=20)

    # -------------------------------------------------------------
    # 1. RIDDLE COUNTS & DIFFICULTIES
    # -------------------------------------------------------------
    def test_01_riddle_counts_by_difficulty(self):
        """Verify exactly 5 riddles per difficulty and 15 overall."""
        self.assertEqual(len(RIDDLES_EASY), 5, "Easy mode must have exactly 5 riddles.")
        self.assertEqual(len(RIDDLES_MEDIUM), 5, "Medium mode must have exactly 5 riddles.")
        self.assertEqual(len(RIDDLES_HARD), 5, "Hard mode must have exactly 5 riddles.")
        self.assertEqual(len(RIDDLES_15), 15, "Total riddles must be exactly 15.")

        for r in RIDDLES_EASY:
            self.assertEqual(r.difficulty, "Easy")
        for r in RIDDLES_MEDIUM:
            self.assertEqual(r.difficulty, "Medium")
        for r in RIDDLES_HARD:
            self.assertEqual(r.difficulty, "Hard")

    def test_02_riddle_uniqueness(self):
        """Verify unique IDs, titles, and riddle texts across all 15 riddles."""
        ids = [r.id for r in RIDDLES_15]
        titles = [r.title for r in RIDDLES_15]
        riddles = [r.scenario for r in RIDDLES_15]

        self.assertEqual(len(ids), len(set(ids)), "All riddle IDs must be unique.")
        self.assertEqual(len(titles), len(set(titles)), "All riddle titles must be unique.")
        self.assertEqual(len(riddles), len(set(riddles)), "All riddle texts must be unique.")

    # -------------------------------------------------------------
    # 2. VALIDATION AGAINST MODEL LABELS
    # -------------------------------------------------------------
    def test_03_riddle_answers_match_label_mapping(self):
        """Verify every riddle answer is a valid class in label_mapping_20.json."""
        valid_classes = set(self.label_mapping_20.keys())

        expected_easy = ["candle", "umbrella", "fish", "bicycle", "book"]
        expected_medium = ["clock", "key", "flower", "shoe", "house"]
        expected_hard = ["scissors", "parachute", "airplane", "tree", "axe"]

        actual_easy = [r.accepted_classes[0] for r in RIDDLES_EASY]
        actual_medium = [r.accepted_classes[0] for r in RIDDLES_MEDIUM]
        actual_hard = [r.accepted_classes[0] for r in RIDDLES_HARD]

        self.assertEqual(actual_easy, expected_easy)
        self.assertEqual(actual_medium, expected_medium)
        self.assertEqual(actual_hard, expected_hard)

        for r in RIDDLES_15:
            self.assertEqual(len(r.accepted_classes), 1, f"Riddle {r.id} should have exactly 1 intended answer.")
            answer = r.accepted_classes[0]
            self.assertIn(answer, valid_classes, f"Riddle answer '{answer}' not in 20-class mapping.")
            self.assertEqual(r.intended_answer, answer)
            self.assertTrue(r.scenario.strip())
            self.assertTrue(r.prompt.strip())
            self.assertTrue(r.success_message.strip())
            self.assertTrue(r.failure_hint.strip())
            self.assertGreaterEqual(len(r.progressive_hints), 2)

    def test_04_validate_challenges_module_function(self):
        """Verify validate_challenges(20) and validate_challenges(8) execute without error."""
        self.assertTrue(validate_challenges(version=8))
        self.assertTrue(validate_challenges(version=20))

    # -------------------------------------------------------------
    # 3. HELPER FUNCTIONS: get_riddles & get_challenges
    # -------------------------------------------------------------
    def test_05_get_riddles_api(self):
        """Verify get_riddles with various arguments and case-insensitivity."""
        easy = get_riddles("easy")
        self.assertEqual(len(easy), 5)
        self.assertEqual([r.id for r in easy], [r.id for r in RIDDLES_EASY])

        medium = get_riddles("Medium")
        self.assertEqual(len(medium), 5)

        hard = get_riddles("HARD")
        self.assertEqual(len(hard), 5)

        all_riddles = get_riddles()
        self.assertEqual(len(all_riddles), 15)

        all_riddles_kw = get_riddles("all")
        self.assertEqual(len(all_riddles_kw), 15)

        with self.assertRaises(ValueError):
            get_riddles("nightmare")

    def test_06_get_challenges_with_difficulty(self):
        """Verify get_challenges accepts difficulty parameter and defaults properly."""
        diff_easy = get_challenges(difficulty="easy")
        self.assertEqual(len(diff_easy), 5)

        diff_med = get_challenges(difficulty="medium")
        self.assertEqual(len(diff_med), 5)

        diff_hard = get_challenges(difficulty="hard")
        self.assertEqual(len(diff_hard), 5)

        # Legacy assertions
        legacy_default = get_challenges()
        self.assertEqual(len(legacy_default), 8)

        version_8 = get_challenges(version=8)
        self.assertEqual(len(version_8), 8)

        version_20 = get_challenges(version=20)
        self.assertEqual(len(version_20), 20)

    # -------------------------------------------------------------
    # 4. GAMEPLAY PROGRESSION & SCORING
    # -------------------------------------------------------------
    def test_07_complete_easy_riddle_progression_and_victory(self):
        """Play through all 5 Easy riddles on first try: verify +150 pts/riddle and victory."""
        engine = GameEngine(challenges=get_riddles("easy"), difficulty="easy")
        engine.start_game()
        self.assertEqual(engine.total_rooms, 5)
        self.assertEqual(engine.lives, 3)

        expected_answers = ["candle", "umbrella", "fish", "bicycle", "book"]

        for idx, ans in enumerate(expected_answers):
            self.assertEqual(engine.current_room_idx, idx)
            ch = engine.get_current_challenge()
            self.assertEqual(ch.accepted_classes[0], ans)

            fb = engine.submit_attempt(ans, confidence=0.95, threshold=0.40)
            self.assertTrue(fb["success"])
            self.assertTrue(fb["is_first_try"])
            self.assertEqual(fb["points_awarded"], 150)
            self.assertEqual(engine.game_state, "ROOM_SUCCESS")

            engine.advance_room()

        self.assertEqual(engine.game_state, "VICTORY")
        summary = engine.get_summary()
        self.assertEqual(summary["rooms_solved"], 5)
        self.assertEqual(summary["total_rooms"], 5)
        self.assertEqual(summary["first_try_bonuses"], 5)
        self.assertEqual(summary["lives"], 3)
        self.assertEqual(summary["difficulty"], "easy")
        # Score: 5 * 100 + 5 * 50 + 3 * 25 = 500 + 250 + 75 = 825
        expected_score = (5 * POINTS_PER_ROOM) + (5 * FIRST_TRY_BONUS) + (3 * BONUS_PER_LIFE)
        self.assertEqual(summary["score"], expected_score)

    def test_08_complete_medium_and_hard_riddle_progressions(self):
        """Verify Medium and Hard modes advance cleanly through all 5 riddles."""
        for diff in ["medium", "hard"]:
            engine = GameEngine(challenges=get_riddles(diff), difficulty=diff)
            engine.start_game()
            self.assertEqual(engine.total_rooms, 5)

            for _ in range(5):
                ch = engine.get_current_challenge()
                fb = engine.submit_attempt(ch.accepted_classes[0], confidence=0.92, threshold=0.40)
                self.assertTrue(fb["success"])
                engine.advance_room()

            self.assertEqual(engine.game_state, "VICTORY")
            self.assertEqual(engine.score, (5 * 150) + (3 * 25))

    def test_09_incorrect_then_correct_attempt_scoring(self):
        """Wrong answer consumes 1 life; subsequent correct attempt awards 100 pts without first-try bonus."""
        engine = GameEngine(challenges=get_riddles("easy"), difficulty="easy")
        engine.start_game()

        # Riddle 1: answer is candle. Submit wrong answer 'shoe'
        fb_fail = engine.submit_attempt("shoe", confidence=0.90, threshold=0.40)
        self.assertFalse(fb_fail["success"])
        self.assertTrue(fb_fail["life_lost"])
        self.assertEqual(engine.lives, 2)
        self.assertEqual(engine.score, 0)
        self.assertEqual(engine.game_state, "PLAYING")

        # Now submit correct answer 'candle'
        fb_win = engine.submit_attempt("candle", confidence=0.90, threshold=0.40)
        self.assertTrue(fb_win["success"])
        self.assertFalse(fb_win["is_first_try"])
        self.assertEqual(fb_win["points_awarded"], 100)
        self.assertEqual(engine.score, 100)
        self.assertEqual(engine.lives, 2)
        self.assertEqual(engine.game_state, "ROOM_SUCCESS")

    def test_10_empty_canvas_safety(self):
        """Blank canvas submission preserves lives and does not increment attempts."""
        engine = GameEngine(challenges=get_riddles("easy"), difficulty="easy")
        engine.start_game()

        fb = engine.submit_attempt(None, confidence=0.0, threshold=0.40, is_empty=True)
        self.assertFalse(fb["success"])
        self.assertFalse(fb["life_lost"])
        self.assertTrue(fb["is_empty"])
        self.assertEqual(engine.lives, 3)
        self.assertEqual(engine.attempts_this_room, 0)
        self.assertEqual(engine.game_state, "PLAYING")

    def test_11_low_confidence_rejection(self):
        """Prediction below confidence cutoff consumes 1 life as unrecognized."""
        engine = GameEngine(challenges=get_riddles("easy"), difficulty="easy")
        engine.start_game()

        # Candle predicted with only 25% confidence (threshold 40%)
        fb = engine.submit_attempt("candle", confidence=0.25, threshold=0.40)
        self.assertFalse(fb["success"])
        self.assertTrue(fb["life_lost"])
        self.assertIn("below 40% threshold", fb["message"])
        self.assertEqual(engine.lives, 2)

    def test_12_game_over_flow(self):
        """Three failed attempts trigger GAME_OVER state."""
        engine = GameEngine(challenges=get_riddles("easy"), difficulty="easy")
        engine.start_game()

        for _ in range(3):
            engine.submit_attempt("unrelated", confidence=0.90, threshold=0.40)

        self.assertEqual(engine.lives, 0)
        self.assertEqual(engine.game_state, "GAME_OVER")
        summary = engine.get_summary()
        self.assertTrue(summary["is_game_over"])
        self.assertEqual(summary["rooms_solved"], 0)

    # -------------------------------------------------------------
    # 5. RESTART, SWITCHING DIFFICULTY & STATE STABILITY
    # -------------------------------------------------------------
    def test_13_restart_and_difficulty_switch(self):
        """Restarting resets state cleanly; switching difficulty loads exact 5 riddles for new difficulty."""
        engine = GameEngine(challenges=get_riddles("easy"), difficulty="easy")
        engine.start_game()
        engine.submit_attempt("candle", confidence=0.95, threshold=0.40)
        self.assertEqual(engine.score, 150)

        # Restart
        engine.restart_game()
        self.assertEqual(engine.game_state, "WELCOME")
        self.assertEqual(engine.score, 0)
        self.assertEqual(engine.lives, 3)
        self.assertEqual(engine.current_room_idx, 0)

        # Switch to Hard mode
        hard_engine = GameEngine(challenges=get_riddles("hard"), difficulty="hard")
        hard_engine.start_game()
        self.assertEqual(hard_engine.total_rooms, 5)
        self.assertEqual(hard_engine.difficulty, "hard")
        ch1 = hard_engine.get_current_challenge()
        self.assertEqual(ch1.id, "riddle_hard_01_scissors")
        self.assertEqual(ch1.accepted_classes, ["scissors"])

    def test_14_idempotency_prevents_duplicate_points(self):
        """Submitting multiple times on a solved room does not award duplicate points."""
        engine = GameEngine(challenges=get_riddles("easy"), difficulty="easy")
        engine.start_game()

        fb1 = engine.submit_attempt("candle", confidence=0.95, threshold=0.40)
        self.assertTrue(fb1["success"])
        self.assertEqual(engine.score, 150)

        # Attempt submit while already in ROOM_SUCCESS
        fb2 = engine.submit_attempt("candle", confidence=0.95, threshold=0.40)
        self.assertFalse(fb2["success"])
        self.assertIn("Cannot submit attempt in game state 'ROOM_SUCCESS'", fb2["message"])
        self.assertEqual(engine.score, 150)

    # -------------------------------------------------------------
    # 6. INFERENCE MODEL COMPATIBILITY
    # -------------------------------------------------------------
    def test_15_model_predicts_riddle_target_class(self):
        """Verify real 20-class CNN correctly outputs logits for riddle classes."""
        dummy_input = torch.randn(1, 1, 28, 28)
        logits = self.predictor.model(dummy_input)
        self.assertEqual(logits.shape, (1, 20))

        # Check each riddle's intended answer exists in predictor class names
        for r in RIDDLES_15:
            ans = r.intended_answer
            self.assertIn(ans, self.predictor.class_names)
            idx = self.predictor.class_names.index(ans)
            self.assertEqual(self.label_mapping_20[ans], idx)


if __name__ == "__main__":
    unittest.main()
