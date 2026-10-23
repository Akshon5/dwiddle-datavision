from typing import List, Dict, Any, Optional
from src.challenges import get_challenges, Challenge

INITIAL_LIVES = 3
POINTS_PER_ROOM = 100
FIRST_TRY_BONUS = 50
BONUS_PER_LIFE = 25
DEFAULT_CONFIDENCE_THRESHOLD = 0.40


class GameEngine:
    """
    Core game engine for 'Draw to Survive'.
    Maintains game state, validates predictions against challenge criteria,
    manages lives, enforces scoring rules, and handles room progression.
    """
    def __init__(
        self,
        challenges: Optional[List[Challenge]] = None,
        initial_lives: int = INITIAL_LIVES,
        difficulty: Optional[str] = None
    ):
        self.challenges = challenges if challenges is not None else get_challenges()
        self.total_rooms = len(self.challenges)
        self.initial_lives = initial_lives
        self.difficulty = difficulty
        
        # State variables
        self.lives = initial_lives
        self.score = 0
        self.current_room_idx = 0
        self.attempts_this_room = 0
        self.solved_room_ids = set()
        self.game_state = "WELCOME"  # WELCOME, PLAYING, ROOM_SUCCESS, GAME_OVER, VICTORY
        self.last_attempt_feedback = {}
        self.history = []
        self.remaining_lives_bonus = 0
        self.first_try_bonuses_earned = 0
        
    def start_game(self):
        """Transitions from WELCOME to PLAYING."""
        self.game_state = "PLAYING"
        self.last_attempt_feedback = {}
        
    def get_current_challenge(self) -> Challenge:
        """Returns the active challenge object."""
        if self.current_room_idx < self.total_rooms:
            return self.challenges[self.current_room_idx]
        return self.challenges[-1]
        
    def submit_attempt(
        self,
        predicted_class: Optional[str],
        confidence: float,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        is_empty: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates a drawing attempt against the current room's accepted classes.
        
        Rules:
        - Empty canvas: no life consumed, prompts to draw.
        - Non-empty canvas with confidence < threshold: consumes 1 life, marked low confidence.
        - Non-empty canvas with predicted_class not in accepted: consumes 1 life, marked incorrect.
        - Non-empty canvas with predicted_class in accepted AND confidence >= threshold:
          awards 100 points (+50 if first try), marks room solved, transitions to ROOM_SUCCESS.
        - If lives reach 0: transitions to GAME_OVER.
        """
        if self.game_state != "PLAYING":
            return {
                "success": False,
                "is_empty": False,
                "message": f"Cannot submit attempt in game state '{self.game_state}'."
            }
            
        current_ch = self.get_current_challenge()
        
        # 1. Handle empty canvas
        if is_empty or predicted_class is None:
            feedback = {
                "success": False,
                "is_empty": True,
                "predicted_class": None,
                "confidence": 0.0,
                "accepted_classes": current_ch.accepted_classes,
                "life_lost": False,
                "message": "Canvas is blank! Draw an object on the canvas before submitting."
            }
            self.last_attempt_feedback = feedback
            return feedback
            
        # 2. Check confidence threshold
        is_confident = (confidence >= threshold)
        is_accepted = (predicted_class in current_ch.accepted_classes)
        
        if is_confident and is_accepted:
            # SUCCESS
            is_first_try = (self.attempts_this_room == 0)
            room_points = 0
            
            # Guard against duplicate point awards
            if current_ch.id not in self.solved_room_ids:
                room_points += POINTS_PER_ROOM
                if is_first_try:
                    room_points += FIRST_TRY_BONUS
                    self.first_try_bonuses_earned += 1
                self.score += room_points
                self.solved_room_ids.add(current_ch.id)
                
            self.game_state = "ROOM_SUCCESS"
            
            feedback = {
                "success": True,
                "is_empty": False,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "accepted_classes": current_ch.accepted_classes,
                "is_first_try": is_first_try,
                "points_awarded": room_points,
                "life_lost": False,
                "message": current_ch.success_message,
                "lives_remaining": self.lives,
                "total_score": self.score
            }
            
        elif not is_confident:
            # Low confidence non-empty attempt -> Costs 1 life
            self.lives = max(0, self.lives - 1)
            self.attempts_this_room += 1
            life_lost = True
            
            if self.lives == 0:
                self.game_state = "GAME_OVER"
                
            feedback = {
                "success": False,
                "is_empty": False,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "accepted_classes": current_ch.accepted_classes,
                "life_lost": life_lost,
                "lives_remaining": self.lives,
                "message": (
                    f"The AI could not confidently identify your sketch (predicted '{predicted_class}' "
                    f"at {confidence*100:.1f}%, below {threshold*100:.0f}% threshold). Lost 1 life! "
                    f"{current_ch.failure_hint}"
                )
            }
            
        else:
            # Confident but incorrect class -> Costs 1 life
            self.lives = max(0, self.lives - 1)
            self.attempts_this_room += 1
            life_lost = True
            
            if self.lives == 0:
                self.game_state = "GAME_OVER"
                
            feedback = {
                "success": False,
                "is_empty": False,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "accepted_classes": current_ch.accepted_classes,
                "life_lost": life_lost,
                "lives_remaining": self.lives,
                "message": (
                    f"The AI recognized '{predicted_class}' ({confidence*100:.1f}%), which does not solve this room! "
                    f"Lost 1 life! {current_ch.failure_hint}"
                )
            }
            
        self.last_attempt_feedback = feedback
        self.history.append({
            "room_idx": self.current_room_idx,
            "room_id": current_ch.id,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "success": feedback["success"],
            "life_lost": feedback.get("life_lost", False),
            "score_after": self.score,
            "lives_after": self.lives
        })
        return feedback
        
    def advance_room(self) -> str:
        """
        Advances to the next room after a successful challenge.
        Transitions to VICTORY if all rooms are completed.
        """
        if self.game_state != "ROOM_SUCCESS":
            return self.game_state
            
        self.current_room_idx += 1
        self.attempts_this_room = 0
        self.last_attempt_feedback = {}
        
        if self.current_room_idx >= self.total_rooms:
            # VICTORY!
            self.game_state = "VICTORY"
            # Award remaining lives bonus once
            self.remaining_lives_bonus = self.lives * BONUS_PER_LIFE
            self.score += self.remaining_lives_bonus
        else:
            self.game_state = "PLAYING"
            
        return self.game_state
        
    def restart_game(self):
        """Completely resets game state back to WELCOME screen."""
        self.lives = self.initial_lives
        self.score = 0
        self.current_room_idx = 0
        self.attempts_this_room = 0
        self.solved_room_ids = set()
        self.game_state = "WELCOME"
        self.last_attempt_feedback = {}
        self.history = []
        self.remaining_lives_bonus = 0
        self.first_try_bonuses_earned = 0
        
    def get_summary(self) -> Dict[str, Any]:
        """Returns snapshot of current state for UI rendering or assertion."""
        return {
            "game_state": self.game_state,
            "lives": self.lives,
            "initial_lives": self.initial_lives,
            "score": self.score,
            "current_room_number": min(self.current_room_idx + 1, self.total_rooms),
            "current_room_idx": self.current_room_idx,
            "total_rooms": self.total_rooms,
            "attempts_this_room": self.attempts_this_room,
            "rooms_solved": len(self.solved_room_ids),
            "first_try_bonuses": self.first_try_bonuses_earned,
            "remaining_lives_bonus": self.remaining_lives_bonus,
            "difficulty": self.difficulty,
            "is_game_over": self.game_state == "GAME_OVER",
            "is_victory": self.game_state == "VICTORY"
        }
