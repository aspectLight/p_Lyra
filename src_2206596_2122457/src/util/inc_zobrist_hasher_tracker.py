from typing import Dict, Optional, Tuple
import numpy as np
from seahorse.game.game_layout.board import Piece
from src_2206596_2122457.src.constants import BOARD_SIZE, PIECE_B, PIECE_R
from src_2206596_2122457.src.precomputed.zobrist_constants import (
    PIECE_TO_ZOBRIST_TABLE,
    PIECE_TO_PLAYER_BIT,
)


class Zobrist:
    def __init__(self, board_size: int) -> None:
        # We use precomputed constants for speed and stability. The seed is ignored.
        self.board_size: int = board_size
        self.num_positions: int = board_size * board_size

        # Numpy uint64 tables from precomputed constants
        self.table_b: np.ndarray = PIECE_TO_ZOBRIST_TABLE[PIECE_B]
        self.table_r: np.ndarray = PIECE_TO_ZOBRIST_TABLE[PIECE_R]
        self.player_b: np.uint64 = PIECE_TO_PLAYER_BIT[PIECE_B]
        self.player_r: np.uint64 = PIECE_TO_PLAYER_BIT[PIECE_R]

        # Branchless lookups for hot paths
        self.table_by_piece: Dict[int, np.ndarray] = {
            PIECE_B: self.table_b,
            PIECE_R: self.table_r,
        }
        self.player_by_piece: Dict[int, np.uint64] = {
            PIECE_B: self.player_b,
            PIECE_R: self.player_r,
        }

    def compute_hash_from_array(self, board_array: np.ndarray) -> int:
        # Vectorized XOR across occupied cells for each color
        mask_b = (board_array == PIECE_B)
        mask_r = (board_array == PIECE_R)
        hb = np.bitwise_xor.reduce(self.table_b[mask_b], dtype=np.uint64, initial=np.uint64(0))
        hr = np.bitwise_xor.reduce(self.table_r[mask_r], dtype=np.uint64, initial=np.uint64(0))
        h = int(hb ^ hr)
        return h

    def compute_hash_from_env(self, env: Dict[Tuple[int, int], Piece]) -> int:
        h: int = 0
        for (r, c), piece in env.items():
            t = piece.get_type()  # "R" or "B"
            piece_int = PIECE_R if t == "R" else PIECE_B
            idx = r * self.board_size + c
            h ^= self.table_b[idx] if piece_int == PIECE_B else self.table_r[idx]
        return h

    def xor_player(self, h: int, player_piece: int) -> int:
        return int(np.uint64(h) ^ self.player_by_piece[player_piece])

    def checksum(self) -> int:
        return hash(tuple[int, ...](self.table_b + self.table_r)) & 0xFFFF_FFFF

    def player_bit(self, player_piece: int) -> int:
        return int(self.player_by_piece[player_piece])

    def apply_move_hash(self, current_hash: int, current_player_piece: int, action_id: int) -> tuple[int, int]:
        """
        Pure function to compute next hash after placing current_player_piece at action_id.
        Returns (next_hash, next_player_piece).
        """
        h = int(np.uint64(current_hash) ^ self.player_by_piece[current_player_piece])
        h ^= int(self.table_by_piece[current_player_piece][action_id])
        next_player = PIECE_R if current_player_piece == PIECE_B else PIECE_B
        h ^= int(self.player_by_piece[next_player])
        return int(h), next_player


class IncrementalZobristHashTracker:
    def __init__(self, board_size: int = BOARD_SIZE, seed: Optional[int] = None) -> None:
        self.board_size: int = board_size
        self.zobrist: Zobrist = Zobrist(board_size, seed)
        self.current_hash: int = 0
        # Track the side to move as integer piece code (PIECE_B=1, PIECE_R=2)
        self.current_player_piece: int = PIECE_B

    def initialize_from_array(self, board_array: np.ndarray, next_player_piece: int) -> None:
        self.current_hash = self.zobrist.compute_hash_from_array(board_array)
        self.current_player_piece = int(next_player_piece)
        self.current_hash = self.zobrist.xor_player(self.current_hash, self.current_player_piece)

    def initialize_from_env(self, env: Dict[Tuple[int, int], Piece], next_player_piece: int) -> None:
        self.current_hash = self.zobrist.compute_hash_from_env(env)
        self.current_player_piece = int(next_player_piece)
        self.current_hash = self.zobrist.xor_player(self.current_hash, self.current_player_piece)

    def apply_move(self, action_id: int, piece_type: int) -> None:
        zob = self.zobrist
        h = self.current_hash
        p = self.current_player_piece
        h ^= int(zob.player_by_piece[p])
        h ^= int(zob.table_by_piece[piece_type][action_id])
        p = PIECE_R if p == PIECE_B else PIECE_B
        h ^= int(zob.player_by_piece[p])
        self.current_hash, self.current_player_piece = h, p

    def undo_move(self, action_id: int, piece_type: int) -> None:
        zob = self.zobrist
        h = self.current_hash
        p = self.current_player_piece
        h ^= int(zob.player_by_piece[p])
        p = PIECE_R if p == PIECE_B else PIECE_B
        h ^= int(zob.table_by_piece[piece_type][action_id])
        h ^= int(zob.player_by_piece[p])
        self.current_hash, self.current_player_piece = h, p

    def set_player_to_move(self, player_piece: int) -> None:
        # Re-set player-to-move bit: remove current, set provided
        self.current_hash ^= int(self.zobrist.player_by_piece[self.current_player_piece])
        self.current_player_piece = int(player_piece)
        self.current_hash ^= int(self.zobrist.player_by_piece[self.current_player_piece])

    def get_hash(self) -> int:
        return self.current_hash

    def recompute_from_array(self, board_array: np.ndarray) -> None:
        self.current_hash = self.zobrist.compute_hash_from_array(board_array)
        self.current_hash ^= int(self.zobrist.player_by_piece[self.current_player_piece])

