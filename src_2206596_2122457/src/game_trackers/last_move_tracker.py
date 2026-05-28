from typing import Optional, Tuple, Dict
from seahorse.game.game_layout.board import Piece
from src_2206596_2122457.src.constants import BOARD_SIZE


class LastMoveTracker:
    def __init__(self) -> None:
        self.board_size: int = BOARD_SIZE
        self.last_env: Optional[Dict[Tuple[int, int], Piece]] = None

    def set_our_board(self, env: Dict[Tuple[int, int], Piece]) -> None:
        self.last_env = env

    def compute_opponent_move(
        self, current_env: Dict[Tuple[int, int], Piece], our_player_piece: str
    ) -> Optional[Tuple[int, int]]:
        if self.last_env is None:
            return None
        
        last_env: Dict[Tuple[int, int], Piece] = self.last_env
        
        all_moves = [
            pos for pos, piece in current_env.items()
            if pos not in last_env or last_env[pos] != piece
        ]
        
        opponent_moves = [
            pos for pos in all_moves
            if current_env[pos].get_type() != our_player_piece
        ]
        
        if not opponent_moves:
            return None
        
        return opponent_moves[0]

    def reset(self) -> None:
        self.last_env = None