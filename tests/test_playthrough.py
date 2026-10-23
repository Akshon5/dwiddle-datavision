import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.challenges import get_challenges
from src.game_engine import GameEngine, POINTS_PER_ROOM, FIRST_TRY_BONUS, BONUS_PER_LIFE


class TestFullPlaythrough(unittest.TestCase):
    def test_complete_victory_playthrough(self):
        engine = GameEngine()
        self.assertEqual(engine.game_state, "WELCOME")
        
        engine.start_game()
        self.assertEqual(engine.game_state, "PLAYING")
        self.assertEqual(engine.lives, 3)
        self.assertEqual(engine.score, 0)
        
        # Room 1: The Armored Cell (accepted: key, door)
        ch1 = engine.get_current_challenge()
        self.assertEqual(ch1.id, "room_1_cell")
        res1 = engine.submit_attempt("key", confidence=0.92, threshold=0.40)
        self.assertTrue(res1["success"])
        self.assertEqual(engine.score, 150)  # 100 + 50 first try
        self.assertEqual(engine.game_state, "ROOM_SUCCESS")
        engine.advance_room()
        
        # Room 2: The Collapsed Timberway (accepted: axe)
        ch2 = engine.get_current_challenge()
        self.assertEqual(ch2.id, "room_2_corridor")
        res2 = engine.submit_attempt("axe", confidence=0.88, threshold=0.40)
        self.assertTrue(res2["success"])
        self.assertEqual(engine.score, 300)
        engine.advance_room()
        
        # Room 3: Pitch-Black Basement (accepted: candle)
        # Simulate 1 mistake first
        res3_fail = engine.submit_attempt("door", confidence=0.75, threshold=0.40)
        self.assertFalse(res3_fail["success"])
        self.assertEqual(engine.lives, 2)
        # Now correct
        res3 = engine.submit_attempt("candle", confidence=0.95, threshold=0.40)
        self.assertTrue(res3["success"])
        self.assertEqual(engine.score, 400)  # Only +100 (no first try bonus)
        engine.advance_room()
        
        # Room 4: Ventilation Conduit (accepted: bicycle)
        res4 = engine.submit_attempt("bicycle", confidence=0.80, threshold=0.40)
        self.assertTrue(res4["success"])
        self.assertEqual(engine.score, 550)  # +150
        engine.advance_room()
        
        # Room 5: Medical Quarantine (accepted: apple)
        res5 = engine.submit_attempt("apple", confidence=0.99, threshold=0.40)
        self.assertTrue(res5["success"])
        self.assertEqual(engine.score, 700)  # +150
        engine.advance_room()
        
        # Room 6: Security Checkpoint (accepted: key, door)
        res6 = engine.submit_attempt("door", confidence=0.91, threshold=0.40)
        self.assertTrue(res6["success"])
        self.assertEqual(engine.score, 850)  # +150
        engine.advance_room()
        
        # Room 7: Rooftop Chasm (accepted: parachute)
        res7 = engine.submit_attempt("parachute", confidence=0.87, threshold=0.40)
        self.assertTrue(res7["success"])
        self.assertEqual(engine.score, 1000)  # +150
        engine.advance_room()
        
        # Room 8: Final Runway (accepted: airplane)
        res8 = engine.submit_attempt("airplane", confidence=0.84, threshold=0.40)
        self.assertTrue(res8["success"])
        self.assertEqual(engine.score, 1150)  # +150
        
        # Final room advance triggers VICTORY!
        engine.advance_room()
        self.assertEqual(engine.game_state, "VICTORY")
        # Remaining lives bonus: 2 lives * 25 = 50 pts
        self.assertEqual(engine.remaining_lives_bonus, 50)
        self.assertEqual(engine.score, 1200)
        
    def test_game_over_playthrough(self):
        engine = GameEngine()
        engine.start_game()
        
        # Lose all 3 lives in Room 1
        res1 = engine.submit_attempt("wrong1", 0.80)
        self.assertEqual(engine.lives, 2)
        self.assertEqual(engine.game_state, "PLAYING")
        
        res2 = engine.submit_attempt("wrong2", 0.80)
        self.assertEqual(engine.lives, 1)
        self.assertEqual(engine.game_state, "PLAYING")
        
        res3 = engine.submit_attempt("wrong3", 0.80)
        self.assertEqual(engine.lives, 0)
        self.assertEqual(engine.game_state, "GAME_OVER")
        
        # Further submissions rejected
        res4 = engine.submit_attempt("key", 0.99)
        self.assertFalse(res4["success"])
        
        # Restart works
        engine.restart_game()
        self.assertEqual(engine.game_state, "WELCOME")
        self.assertEqual(engine.lives, 3)
        self.assertEqual(engine.score, 0)


if __name__ == '__main__':
    unittest.main()
