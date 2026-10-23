"""
Unit and integration tests for the expanded 20-class CNN and 20-room adventure.
Tests checkpoint loading, 20-logit outputs, label mappings, challenges integrity,
and game engine state machine under the expanded mode.
"""

import json
import os
import unittest
import numpy as np
import torch

from src.challenges import CHALLENGES_20, CHALLENGES_8, get_challenges
from src.game_engine import GameEngine, POINTS_PER_ROOM, FIRST_TRY_BONUS, BONUS_PER_LIFE
from src.inference import QuickDrawPredictor
from src.model import QuickDrawCNN


class TestExpanded20Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = os.path.join("models", "expanded_cnn_20.pt")
        cls.label_path = os.path.join("models", "label_mapping_20.json")
        cls.data_label_path = os.path.join("data", "expanded_20", "label_mapping.json")

        with open(cls.label_path, "r") as f:
            cls.label_mapping = json.load(f)

        cls.predictor = QuickDrawPredictor(model_version=20)
        cls.challenges = get_challenges(20)

    def test_01_label_mappings_consistency(self):
        """Verify both 20-class label files exist, match each other, and have 20 classes."""
        self.assertTrue(os.path.exists(self.label_path))
        self.assertTrue(os.path.exists(self.data_label_path))

        with open(self.data_label_path, "r") as f:
            data_mapping = json.load(f)

        self.assertEqual(self.label_mapping, data_mapping)
        self.assertEqual(len(self.label_mapping), 20)

        expected_classes = sorted([
            "airplane", "apple", "axe", "bicycle", "book",
            "candle", "chair", "clock", "cup", "door",
            "fish", "flower", "house", "key", "parachute",
            "scissors", "shoe", "star", "tree", "umbrella"
        ])
        actual_classes = sorted(list(self.label_mapping.keys()))
        self.assertEqual(actual_classes, expected_classes)

        # Ensure indices are 0 to 19 contiguous
        indices = sorted(list(self.label_mapping.values()))
        self.assertEqual(indices, list(range(20)))

    def test_02_model_checkpoint_and_architecture(self):
        """Verify checkpoint loads into QuickDrawCNN(num_classes=20) and weights match."""
        self.assertTrue(os.path.exists(self.checkpoint_path))
        model = QuickDrawCNN(num_classes=20)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        self.assertEqual(model.classifier[5].out_features, 20)
        self.assertEqual(model.classifier[2].in_features, 128 * 3 * 3)

    def test_03_inference_forward_logits_shape(self):
        """Verify model output shape is strictly (batch_size, 20) and probabilities sum to 1."""
        dummy_input = torch.randn(4, 1, 28, 28)
        logits = self.predictor.model(dummy_input)
        self.assertEqual(logits.shape, (4, 20))

        probs = torch.softmax(logits, dim=1)
        self.assertEqual(probs.shape, (4, 20))
        for row in probs:
            self.assertAlmostEqual(float(row.sum().item()), 1.0, places=4)

    def test_04_predictor_class_names(self):
        """Verify predictor exposes exactly 20 class names matching index order."""
        self.assertEqual(self.predictor.num_classes, 20)
        self.assertEqual(len(self.predictor.class_names), 20)
        for name, idx in self.label_mapping.items():
            self.assertEqual(self.predictor.class_names[idx], name)

    def test_05_predict_canvas_structure(self):
        """Verify predict_canvas returns class, confidence, top3, and meta dict."""
        canvas = np.ones((300, 300, 4), dtype=np.uint8) * 255
        # Draw a synthetic black line in the middle
        canvas[140:160, 100:200, :3] = 0

        pred_class, conf, top3, meta = self.predictor.predict_canvas(canvas)
        self.assertIn(pred_class, self.predictor.class_names)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertEqual(len(top3), 3)
        self.assertIn("processed_28x28", meta)
        self.assertIn("is_empty", meta)
        self.assertFalse(meta["is_empty"])

    def test_06_predict_canvas_empty_canvas(self):
        """Verify empty canvas is flagged in metadata."""
        empty_canvas = np.ones((300, 300, 4), dtype=np.uint8) * 255
        pred_class, conf, top3, meta = self.predictor.predict_canvas(empty_canvas)
        self.assertTrue(meta.get("is_empty", False))

    def test_07_challenges_20_count_and_uniqueness(self):
        """Verify there are exactly 20 challenges with unique IDs and titles."""
        self.assertEqual(len(self.challenges), 20)
        self.assertEqual(len(CHALLENGES_20), 20)

        ids = [c.id for c in self.challenges]
        titles = [c.title for c in self.challenges]
        self.assertEqual(len(ids), len(set(ids)), "Challenge IDs must be unique.")
        self.assertEqual(len(titles), len(set(titles)), "Challenge titles must be unique.")

    def test_08_challenges_accepted_classes_validity(self):
        """Verify every accepted class in all 20 challenges exists in the 20-class mapping."""
        valid_classes = set(self.label_mapping.keys())
        for ch in self.challenges:
            self.assertGreater(len(ch.accepted_classes), 0, f"Challenge {ch.id} has no accepted classes")
            for ac in ch.accepted_classes:
                self.assertIn(
                    ac, valid_classes,
                    f"Challenge {ch.id} accepted class '{ac}' not in 20-class vocabulary!"
                )
            self.assertTrue(ch.scenario.strip())
            self.assertTrue(ch.prompt.strip())
            self.assertTrue(ch.success_message.strip())
            self.assertTrue(ch.failure_hint.strip())
            self.assertIn(ch.difficulty, ["Introductory", "Moderate", "Challenging", "Easy", "Medium", "Hard", "Expert", "Nightmare"])
            self.assertIsInstance(ch.progressive_hints, list)
            self.assertGreaterEqual(len(ch.progressive_hints), 2, f"Challenge {ch.id} missing progressive hints")

    def test_09_all_20_classes_represented_across_challenges(self):
        """Verify that across the 20 challenges, all 20 categories can be utilized."""
        represented_classes = set()
        for ch in self.challenges:
            for ac in ch.accepted_classes:
                represented_classes.add(ac)
        missing = set(self.label_mapping.keys()) - represented_classes
        self.assertEqual(len(missing), 0, f"Classes not utilized in any challenge: {missing}")

    def test_10_game_engine_initialization_20_rooms(self):
        """Verify GameEngine initializes properly for 20 rooms."""
        engine = GameEngine(challenges=self.challenges)
        self.assertEqual(engine.total_rooms, 20)
        self.assertEqual(engine.lives, 3)
        self.assertEqual(engine.score, 0)
        self.assertEqual(engine.game_state, "WELCOME")

        engine.start_game()
        self.assertEqual(engine.game_state, "PLAYING")
        self.assertEqual(engine.current_room_idx, 0)

    def test_11_game_engine_scoring_and_first_try_bonus(self):
        """Verify scoring logic on first-try vs retry."""
        engine = GameEngine(challenges=self.challenges)
        engine.start_game()

        ch0 = engine.get_current_challenge()
        accepted_item = ch0.accepted_classes[0]

        # Room 1: First try success -> +100 base + 50 first try = 150
        fb = engine.submit_attempt(accepted_item, 0.95, threshold=0.40)
        self.assertTrue(fb["success"])
        self.assertEqual(fb["points_awarded"], 150)
        self.assertTrue(fb["is_first_try"])
        self.assertEqual(engine.score, 150)

        # Advance to Room 2
        engine.advance_room()
        self.assertEqual(engine.current_room_idx, 1)

        # Room 2: First attempt wrong -> lose 1 life, score unchanged
        fb_fail = engine.submit_attempt("unrelated_item", 0.90, threshold=0.40)
        self.assertFalse(fb_fail["success"])
        self.assertTrue(fb_fail["life_lost"])
        self.assertEqual(engine.lives, 2)
        self.assertEqual(engine.score, 150)

        # Room 2: Second attempt success -> +100 base, 0 first try = 100
        ch1 = engine.get_current_challenge()
        accepted_item_2 = ch1.accepted_classes[0]
        fb_success = engine.submit_attempt(accepted_item_2, 0.95, threshold=0.40)
        self.assertTrue(fb_success["success"])
        self.assertEqual(fb_success["points_awarded"], 100)
        self.assertFalse(fb_success["is_first_try"])
        self.assertEqual(engine.score, 250)

    def test_12_game_engine_empty_canvas_safety(self):
        """Verify blank canvas does not deduct a life."""
        engine = GameEngine(challenges=self.challenges)
        engine.start_game()

        fb = engine.submit_attempt("shoe", 0.99, threshold=0.40, is_empty=True)
        self.assertFalse(fb["success"])
        self.assertFalse(fb["life_lost"])
        self.assertTrue(fb["is_empty"])
        self.assertEqual(engine.lives, 3)

    def test_13_game_engine_below_threshold_rejection(self):
        """Verify below-threshold prediction is rejected and costs a life."""
        engine = GameEngine(challenges=self.challenges)
        engine.start_game()

        ch0 = engine.get_current_challenge()
        accepted_item = ch0.accepted_classes[0]

        fb = engine.submit_attempt(accepted_item, 0.25, threshold=0.40)
        self.assertFalse(fb["success"])
        self.assertTrue(fb["life_lost"])
        self.assertIn("below 40% threshold", fb["message"])
        self.assertEqual(engine.lives, 2)

    def test_14_game_engine_game_over_flow(self):
        """Verify losing 3 lives transitions to GAME_OVER."""
        engine = GameEngine(challenges=self.challenges)
        engine.start_game()

        for i in range(3):
            engine.submit_attempt("unrelated", 0.90, threshold=0.40)

        self.assertEqual(engine.lives, 0)
        self.assertEqual(engine.game_state, "GAME_OVER")

    def test_15_game_engine_full_20_room_victory(self):
        """Simulate solving all 20 rooms with 3 lives intact to verify VICTORY state & score."""
        engine = GameEngine(challenges=self.challenges)
        engine.start_game()

        for room_idx in range(20):
            current_ch = engine.get_current_challenge()
            solution = current_ch.accepted_classes[0]
            fb = engine.submit_attempt(solution, 0.95, threshold=0.40)
            self.assertTrue(fb["success"], f"Failed to solve room {room_idx}: {current_ch.id}")
            self.assertEqual(engine.game_state, "ROOM_SUCCESS")
            engine.advance_room()

        self.assertEqual(engine.game_state, "VICTORY")
        summary = engine.get_summary()
        self.assertEqual(summary["rooms_solved"], 20)
        self.assertEqual(summary["total_rooms"], 20)
        self.assertEqual(summary["first_try_bonuses"], 20)
        self.assertEqual(summary["lives"], 3)
        expected_score = (20 * POINTS_PER_ROOM) + (20 * FIRST_TRY_BONUS) + (3 * BONUS_PER_LIFE)
        self.assertEqual(summary["score"], expected_score)

    def test_16_both_versions_coexist_independently(self):
        """Verify 8-class and 20-class predictors and challenges coexist without collision."""
        pred_8 = QuickDrawPredictor(model_version=8)
        pred_20 = QuickDrawPredictor(model_version=20)
        ch_8 = get_challenges(8)
        ch_20 = get_challenges(20)

        self.assertEqual(pred_8.num_classes, 8)
        self.assertEqual(pred_20.num_classes, 20)
        self.assertEqual(len(ch_8), 8)
        self.assertEqual(len(ch_20), 20)


if __name__ == "__main__":
    unittest.main()
