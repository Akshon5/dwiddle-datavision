import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.challenges import get_challenges, get_challenge
from src.game_engine import GameEngine, POINTS_PER_ROOM, FIRST_TRY_BONUS, BONUS_PER_LIFE


class TestGameStateExhaustive(unittest.TestCase):
    def setUp(self):
        self.challenges = get_challenges()
        self.engine = GameEngine()
        self.engine.start_game()

    def test_all_eight_challenges_individually(self):
        """Test each of the 8 challenges has valid title, narrative, prompt, and accepted classes."""
        self.assertEqual(len(self.challenges), 8)
        
        expected_accepted = {
            "room_1_cell": ["key", "door"],
            "room_2_corridor": ["axe"],
            "room_3_blackout": ["candle"],
            "room_4_tunnel": ["bicycle"],
            "room_5_quarantine": ["apple"],
            "room_6_airlock": ["key", "door"],
            "room_7_rooftop": ["parachute"],
            "room_8_runway": ["airplane"]
        }
        
        for idx, ch in enumerate(self.challenges):
            self.assertTrue(ch.title.startswith(f"Room {idx+1}:"), f"Room {idx+1} title mismatch")
            self.assertGreater(len(ch.scenario), 20, f"Scenario too short for {ch.id}")
            self.assertTrue(ch.prompt.startswith("Draw "), f"Prompt doesn't start with 'Draw' for {ch.id}")
            self.assertEqual(ch.accepted_classes, expected_accepted[ch.id])
            self.assertGreater(len(ch.success_message), 10)
            self.assertGreater(len(ch.failure_hint), 10)

    def test_room_by_room_sequential_solving(self):
        """Solve each room step-by-step and verify points and state at every stage."""
        total_expected_score = 0
        
        for idx in range(8):
            self.assertEqual(self.engine.current_room_idx, idx)
            self.assertEqual(self.engine.game_state, "PLAYING")
            
            ch = self.engine.get_current_challenge()
            solution = ch.accepted_classes[0]
            
            # Submit solution
            res = self.engine.submit_attempt(solution, confidence=0.88, threshold=0.40)
            self.assertTrue(res["success"])
            self.assertTrue(res["is_first_try"])
            self.assertEqual(self.engine.game_state, "ROOM_SUCCESS")
            
            total_expected_score += (POINTS_PER_ROOM + FIRST_TRY_BONUS)
            self.assertEqual(self.engine.score, total_expected_score)
            
            # Advance room
            new_state = self.engine.advance_room()
            if idx < 7:
                self.assertEqual(new_state, "PLAYING")
                self.assertEqual(self.engine.current_room_idx, idx + 1)
            else:
                self.assertEqual(new_state, "VICTORY")
                total_expected_score += (3 * BONUS_PER_LIFE)
                self.assertEqual(self.engine.score, total_expected_score)

    def test_repeated_advance_room_calls_are_safe(self):
        """Verify calling advance_room repeatedly while in PLAYING does not skip rooms."""
        self.assertEqual(self.engine.current_room_idx, 0)
        self.assertEqual(self.engine.game_state, "PLAYING")
        
        # Advance while in PLAYING state (not ROOM_SUCCESS) -> should be a no-op!
        res_state = self.engine.advance_room()
        self.assertEqual(res_state, "PLAYING")
        self.assertEqual(self.engine.current_room_idx, 0)

    def test_repeated_submissions_in_room_success_cannot_award_points(self):
        """Verify repeated clicks in ROOM_SUCCESS cannot duplicate score or change state."""
        ch = self.engine.get_current_challenge()
        res1 = self.engine.submit_attempt(ch.accepted_classes[0], 0.90)
        self.assertTrue(res1["success"])
        score_after_solve = self.engine.score
        
        # Click submit 5 more times
        for _ in range(5):
            res_repeat = self.engine.submit_attempt(ch.accepted_classes[0], 0.90)
            self.assertFalse(res_repeat["success"])
            self.assertEqual(self.engine.score, score_after_solve)
            self.assertEqual(self.engine.game_state, "ROOM_SUCCESS")

    def test_last_life_boundary_to_game_over(self):
        """Verify when lives=1, a single wrong attempt moves lives to 0 and state to GAME_OVER."""
        # Fail 2 times
        self.engine.submit_attempt("wrong1", 0.90)
        self.assertEqual(self.engine.lives, 2)
        self.assertEqual(self.engine.game_state, "PLAYING")
        
        self.engine.submit_attempt("wrong2", 0.90)
        self.assertEqual(self.engine.lives, 1)
        self.assertEqual(self.engine.game_state, "PLAYING")
        
        # Boundary: 1 life left -> failed attempt
        res_fatal = self.engine.submit_attempt("wrong3", 0.90)
        self.assertFalse(res_fatal["success"])
        self.assertTrue(res_fatal["life_lost"])
        self.assertEqual(self.engine.lives, 0)
        self.assertEqual(self.engine.game_state, "GAME_OVER")

    def test_low_confidence_boundary_at_threshold(self):
        """Verify exact threshold boundary: 0.399 is rejected, 0.400 is accepted."""
        ch = self.engine.get_current_challenge()
        correct_class = ch.accepted_classes[0]
        
        # 0.399 -> Below 0.40 threshold -> Rejected, costs 1 life
        res_below = self.engine.submit_attempt(correct_class, 0.399, threshold=0.40)
        self.assertFalse(res_below["success"])
        self.assertTrue(res_below["life_lost"])
        self.assertEqual(self.engine.lives, 2)
        self.assertEqual(self.engine.game_state, "PLAYING")
        
        # 0.400 -> Equal to threshold -> Accepted!
        res_at_thresh = self.engine.submit_attempt(correct_class, 0.400, threshold=0.40)
        self.assertTrue(res_at_thresh["success"])
        self.assertEqual(self.engine.game_state, "ROOM_SUCCESS")

    def test_full_reset_after_game_over(self):
        """Verify restart completely clears failure state, restores lives, and resets history."""
        # Drive to GAME_OVER
        for _ in range(3):
            self.engine.submit_attempt("wrong", 0.90)
        self.assertEqual(self.engine.game_state, "GAME_OVER")
        
        # Restart
        self.engine.restart_game()
        self.assertEqual(self.engine.game_state, "WELCOME")
        self.assertEqual(self.engine.lives, 3)
        self.assertEqual(self.engine.score, 0)
        self.assertEqual(self.engine.current_room_idx, 0)
        self.assertEqual(len(self.engine.solved_room_ids), 0)
        self.assertEqual(len(self.engine.history), 0)


if __name__ == '__main__':
    unittest.main()
