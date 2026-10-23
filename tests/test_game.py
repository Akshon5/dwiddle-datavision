import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.challenges import get_challenges, validate_challenges, Challenge
from src.game_engine import GameEngine, POINTS_PER_ROOM, FIRST_TRY_BONUS, BONUS_PER_LIFE
from src.inference import QuickDrawPredictor
from src.preprocess import preprocess_canvas_image


class TestGameEngine(unittest.TestCase):
    def setUp(self):
        self.challenges = get_challenges()
        self.engine = GameEngine(challenges=self.challenges)
        self.engine.start_game()

    # 1. Correct accepted class solves a challenge
    def test_correct_accepted_class_solves_challenge(self):
        current_ch = self.engine.get_current_challenge()
        accepted_class = current_ch.accepted_classes[0]
        
        result = self.engine.submit_attempt(
            predicted_class=accepted_class,
            confidence=0.85,
            threshold=0.40,
            is_empty=False
        )
        self.assertTrue(result["success"])
        self.assertEqual(self.engine.game_state, "ROOM_SUCCESS")
        self.assertEqual(self.engine.score, POINTS_PER_ROOM + FIRST_TRY_BONUS)
        self.assertEqual(self.engine.lives, 3)

    # 2. Incorrect class fails the challenge and costs a life
    def test_incorrect_class_fails_and_costs_life(self):
        current_ch = self.engine.get_current_challenge()
        # Pick a class definitely not in accepted_classes
        wrong_class = "bicycle" if "bicycle" not in current_ch.accepted_classes else "airplane"
        
        result = self.engine.submit_attempt(
            predicted_class=wrong_class,
            confidence=0.85,
            threshold=0.40,
            is_empty=False
        )
        self.assertFalse(result["success"])
        self.assertTrue(result["life_lost"])
        self.assertEqual(self.engine.lives, 2)
        self.assertEqual(self.engine.game_state, "PLAYING")
        self.assertEqual(self.engine.attempts_this_room, 1)

    # 3. Multiple accepted classes are handled correctly
    def test_multiple_accepted_classes_handled(self):
        # Room 1 accepts both "key" and "door"
        ch = self.engine.get_current_challenge()
        self.assertGreaterEqual(len(ch.accepted_classes), 2)
        
        # Test solution with first accepted class
        res1 = self.engine.submit_attempt("key", 0.90, threshold=0.40)
        self.assertTrue(res1["success"])
        
        # Reset and test with second accepted class
        self.engine.restart_game()
        self.engine.start_game()
        res2 = self.engine.submit_attempt("door", 0.90, threshold=0.40)
        self.assertTrue(res2["success"])

    # 4. Low-confidence nonempty drawings fail and cost one life
    def test_low_confidence_fails_and_costs_life(self):
        current_ch = self.engine.get_current_challenge()
        accepted_class = current_ch.accepted_classes[0]
        
        # Drawing matches accepted class but confidence is below threshold (0.35 < 0.40)
        result = self.engine.submit_attempt(
            predicted_class=accepted_class,
            confidence=0.35,
            threshold=0.40,
            is_empty=False
        )
        self.assertFalse(result["success"])
        self.assertTrue(result["life_lost"])
        self.assertEqual(self.engine.lives, 2)
        self.assertEqual(self.engine.game_state, "PLAYING")

    # 5. Empty canvas does not cost a life
    def test_empty_canvas_does_not_cost_life(self):
        initial_lives = self.engine.lives
        result = self.engine.submit_attempt(
            predicted_class=None,
            confidence=0.0,
            threshold=0.40,
            is_empty=True
        )
        self.assertFalse(result["success"])
        self.assertFalse(result["life_lost"])
        self.assertEqual(self.engine.lives, initial_lives)
        self.assertEqual(self.engine.attempts_this_room, 0)
        self.assertEqual(self.engine.game_state, "PLAYING")

    # 6. A solved challenge cannot award points twice
    def test_solved_challenge_cannot_award_points_twice(self):
        ch = self.engine.get_current_challenge()
        # First solve
        res1 = self.engine.submit_attempt(ch.accepted_classes[0], 0.90)
        initial_score = self.engine.score
        
        # Attempt to submit again in ROOM_SUCCESS
        res2 = self.engine.submit_attempt(ch.accepted_classes[0], 0.90)
        self.assertFalse(res2["success"])
        self.assertEqual(self.engine.score, initial_score)

    # 7. First-attempt bonus is awarded only when appropriate
    def test_first_attempt_bonus_only_on_first_attempt(self):
        ch = self.engine.get_current_challenge()
        wrong_class = "bicycle" if "bicycle" not in ch.accepted_classes else "airplane"
        
        # Attempt 1: Failed
        self.engine.submit_attempt(wrong_class, 0.90)
        self.assertEqual(self.engine.score, 0)
        self.assertEqual(self.engine.attempts_this_room, 1)
        
        # Attempt 2: Correct
        res = self.engine.submit_attempt(ch.accepted_classes[0], 0.90)
        self.assertTrue(res["success"])
        # Awarded POINTS_PER_ROOM (100) but NO first-try bonus (+50)
        self.assertEqual(self.engine.score, POINTS_PER_ROOM)
        self.assertFalse(res["is_first_try"])

    # 8. Lives never become negative
    def test_lives_never_become_negative(self):
        # 10 failed attempts with 3 initial lives
        for _ in range(10):
            self.engine.submit_attempt("wrong_class", 0.90)
        self.assertEqual(self.engine.lives, 0)
        self.assertGreaterEqual(self.engine.lives, 0)

    # 9. The game ends when lives reach zero
    def test_game_over_when_lives_reach_zero(self):
        self.engine.submit_attempt("wrong", 0.90)  # lives = 2
        self.assertEqual(self.engine.game_state, "PLAYING")
        self.engine.submit_attempt("wrong", 0.90)  # lives = 1
        self.assertEqual(self.engine.game_state, "PLAYING")
        self.engine.submit_attempt("wrong", 0.90)  # lives = 0
        self.assertEqual(self.engine.game_state, "GAME_OVER")
        self.assertEqual(self.engine.lives, 0)

    # 10. Completing all challenges triggers victory and awards life bonus
    def test_completing_all_challenges_triggers_victory(self):
        for idx in range(len(self.challenges)):
            ch = self.engine.get_current_challenge()
            self.engine.submit_attempt(ch.accepted_classes[0], 0.90)
            self.assertEqual(self.engine.game_state, "ROOM_SUCCESS")
            self.engine.advance_room()
            
        self.assertEqual(self.engine.game_state, "VICTORY")
        # 8 rooms * (100 + 50 bonus) + (3 lives * 25 bonus) = 8 * 150 + 75 = 1275 points
        expected_score = 8 * (POINTS_PER_ROOM + FIRST_TRY_BONUS) + (3 * BONUS_PER_LIFE)
        self.assertEqual(self.engine.score, expected_score)
        self.assertEqual(self.engine.remaining_lives_bonus, 3 * BONUS_PER_LIFE)

    # 11. Restart restores the initial state
    def test_restart_restores_initial_state(self):
        # Play partially
        ch = self.engine.get_current_challenge()
        self.engine.submit_attempt(ch.accepted_classes[0], 0.90)
        self.engine.advance_room()
        self.assertGreater(self.engine.score, 0)
        
        # Restart
        self.engine.restart_game()
        summary = self.engine.get_summary()
        self.assertEqual(summary["game_state"], "WELCOME")
        self.assertEqual(summary["lives"], 3)
        self.assertEqual(summary["score"], 0)
        self.assertEqual(summary["current_room_idx"], 0)
        self.assertEqual(summary["attempts_this_room"], 0)
        self.assertEqual(summary["rooms_solved"], 0)


class TestIntegrationAndModel(unittest.TestCase):
    # 12. Every accepted class exists in the model label mapping
    def test_all_accepted_classes_exist_in_label_mapping(self):
        self.assertTrue(validate_challenges())

    # 13. Prediction uses the actual trained checkpoint
    def test_prediction_uses_actual_trained_checkpoint(self):
        predictor = QuickDrawPredictor()
        self.assertEqual(len(predictor.class_names), 8)
        
        # Create a synthetic white canvas with a black drawn circle (Apple)
        canvas = np.full((300, 300, 4), 255, dtype=np.uint8)
        # Draw circle
        y, x = np.ogrid[:300, :300]
        mask = (x - 150)**2 + (y - 150)**2 <= 60**2
        canvas[mask] = [0, 0, 0, 255]
        
        pred_class, conf, top3, meta = predictor.predict_canvas(canvas)
        self.assertIsNotNone(pred_class)
        self.assertIn(pred_class, predictor.class_names)
        self.assertGreater(conf, 0.0)
        self.assertEqual(len(top3), 3)
        self.assertFalse(meta["is_empty"])

    # 14. An invalid or malformed canvas image is handled gracefully
    def test_malformed_canvas_handled_gracefully(self):
        # Test None
        t1, m1 = preprocess_canvas_image(None)
        self.assertIsNone(t1)
        self.assertTrue(m1["is_empty"])
        
        # Test 1D array
        t2, m2 = preprocess_canvas_image(np.array([1, 2, 3]))
        self.assertIsNone(t2)
        self.assertTrue(m2["is_empty"])
        
        # Test solid blank canvas
        blank = np.full((100, 100, 4), 255, dtype=np.uint8)
        t3, m3 = preprocess_canvas_image(blank)
        self.assertIsNone(t3)
        self.assertTrue(m3["is_empty"])


if __name__ == '__main__':
    unittest.main()
