import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.game_engine import GameEngine
from src.inference import QuickDrawPredictor


class TestCanvasIntegration(unittest.TestCase):
    def setUp(self):
        self.predictor = QuickDrawPredictor()
        self.engine = GameEngine()
        self.engine.start_game()

    def test_empty_canvas_preserves_lives_in_room_1(self):
        """Verify an empty canvas does not cost a life and prompts user."""
        # Simulated empty canvas: 300x300 solid white RGBA
        empty_canvas = np.full((300, 300, 4), 255, dtype=np.uint8)
        
        pred_class, conf, top3, meta = self.predictor.predict_canvas(empty_canvas)
        self.assertIsNone(pred_class)
        self.assertTrue(meta["is_empty"])
        
        # Submit to engine
        initial_lives = self.engine.lives
        feedback = self.engine.submit_attempt(
            predicted_class=pred_class,
            confidence=conf,
            threshold=0.40,
            is_empty=meta["is_empty"]
        )
        
        self.assertFalse(feedback["success"])
        self.assertFalse(feedback["life_lost"])
        self.assertEqual(self.engine.lives, initial_lives)
        self.assertEqual(self.engine.attempts_this_room, 0)
        self.assertEqual(self.engine.score, 0)
        self.assertEqual(self.engine.game_state, "PLAYING")

    def test_draw_and_submit_key_solves_room_1_and_awards_points_once(self):
        """Draw and submit a key in Room 1: verify CNN runs, solves room, and awards points once."""
        # Check that Room 1 accepts key
        ch = self.engine.get_current_challenge()
        self.assertEqual(ch.id, "room_1_cell")
        self.assertIn("key", ch.accepted_classes)
        
        # 1. Create a synthetic human-drawn key on a 300x300 white canvas
        img = Image.new('RGBA', (300, 300), (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        # Circular head
        d.ellipse([50, 120, 110, 180], outline=(0, 0, 0, 255), width=6)
        # Inner hole
        d.ellipse([70, 140, 90, 160], outline=(0, 0, 0, 255), width=4)
        # Long shaft
        d.line([(110, 150), (240, 150)], fill=(0, 0, 0, 255), width=7)
        # Tooth 1
        d.line([(210, 150), (210, 180)], fill=(0, 0, 0, 255), width=6)
        # Tooth 2
        d.line([(235, 150), (235, 175)], fill=(0, 0, 0, 255), width=6)
        
        canvas_array = np.array(img)
        self.assertEqual(canvas_array.shape, (300, 300, 4))
        
        # 2. Run actual model inference on CPU
        pred_class, conf, top3, meta = self.predictor.predict_canvas(canvas_array)
        
        print(f"\n[Test Canvas Prediction] Predicted: {pred_class} ({conf*100:.2f}%)")
        print(f"[Top-3 Predictions]: {top3}")
        
        self.assertEqual(pred_class, "key")
        self.assertGreater(conf, 0.40)  # High confidence
        self.assertFalse(meta["is_empty"])
        self.assertEqual(len(top3), 3)
        
        # 3. Submit attempt to game engine
        feedback = self.engine.submit_attempt(
            predicted_class=pred_class,
            confidence=conf,
            threshold=0.40,
            is_empty=meta["is_empty"]
        )
        
        self.assertTrue(feedback["success"])
        self.assertEqual(self.engine.game_state, "ROOM_SUCCESS")
        self.assertEqual(self.engine.score, 150)  # 100 room points + 50 first-try bonus
        self.assertEqual(self.engine.lives, 3)
        
        # 4. Verify duplicate submissions cannot award points again
        res2 = self.engine.submit_attempt(
            predicted_class=pred_class,
            confidence=conf,
            threshold=0.40,
            is_empty=False
        )
        self.assertFalse(res2["success"])
        self.assertEqual(self.engine.score, 150)  # Score remains strictly 150


if __name__ == '__main__':
    unittest.main()

