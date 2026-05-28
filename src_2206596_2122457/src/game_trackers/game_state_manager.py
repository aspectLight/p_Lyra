from typing import Optional, Tuple, Dict
from seahorse.game.game_layout.board import Piece
from src_2206596_2122457.src.util.board_utils import action_to_id
from src_2206596_2122457.src.game_trackers.inc_empty_cells_tracker import IncrementalEmptyCellsTracker
from src_2206596_2122457.src.game_trackers.last_move_tracker import LastMoveTracker
from src_2206596_2122457.src.game_trackers.player_identity import PlayerIdentity

class GameStateManager:
    def __init__(self) -> None:
        self.empty_cells_tracker = IncrementalEmptyCellsTracker()
        self.last_move_tracker = LastMoveTracker()
    
    def sync_with_opponent_move(
        self, env: Dict[Tuple[int, int], Piece], player_identity: PlayerIdentity
    ) -> None:
        our_piece_letter = player_identity.get_our_piece_letter()

        opponent_last_move = self.last_move_tracker.compute_opponent_move(
            env,
            our_piece_letter,
        )

        if self.last_move_tracker.last_env is None:
            self._reset_trackers(env)
        else:
            self._apply_opponent_move(opponent_last_move)
        
    def _reset_trackers(self, env: Dict[Tuple[int, int], Piece]) -> None:
        self.empty_cells_tracker.reset_from_env(env)
        self.last_move_tracker.set_our_board(env)

    def _apply_opponent_move(
        self, opponent_last_move: Optional[Tuple[int, int]]
    ) -> None:
        if opponent_last_move is None:
            return
        
        position = opponent_last_move
        last_move_id = action_to_id(position)
        
        self.empty_cells_tracker.remove(last_move_id)
