from typing import Optional

from src_2206596_2122457.src.config import MCTSConfig
from src_2206596_2122457.src.mcts.flat_tree import FlatTree
from src_2206596_2122457.src.util.inc_zobrist_hasher_tracker import Zobrist
from src_2206596_2122457.src.constants import BOARD_SIZE


class TreeManager:
    def __init__(self, config: MCTSConfig):
        self.config = config
        self.flat_tree = FlatTree(
            use_rave=config.use_rave,
            rave_prior_count=config.rave_prior_count,
            rave_prior_value=config.rave_prior_value
        )
        self.root_index: Optional[int] = None
        self.zobrist = Zobrist(BOARD_SIZE)

    def initialize_root(self, root_hash: int, player_piece: int) -> None:
        self.root_index = self.flat_tree.create_root(root_hash, player_piece)

    def advance_on_opponent_move(
        self, opponent_action_id: Optional[int], current_root_hash: int
    ) -> None:
        if opponent_action_id is None or not self.config.use_tree_reuse or self.root_index is None:
            self._reset_tree(current_root_hash, None)
            return
        child_idx = int(self.flat_tree.children[self.root_index, opponent_action_id])
        if child_idx != -1:
            self._advance_root(child_idx)
        else:
            self._reset_tree(current_root_hash, int(self.flat_tree.player_piece[self.root_index]))

    def advance_on_our_move(self, our_action_id: int, current_root_hash: int) -> None:
        if self.root_index is None:
            self._reset_tree(current_root_hash, None)
            return

        child_idx = int(self.flat_tree.children[self.root_index, our_action_id])
        if child_idx != -1:
            self._advance_root(child_idx)
        else:
            self._reset_tree(current_root_hash, int(self.flat_tree.player_piece[self.root_index]))

    def get_root(self) -> Optional[int]:
        return self.root_index

    def get_flat_tree(self) -> FlatTree:
        return self.flat_tree

    def get_root_hash(self) -> int:
        return int(self.flat_tree.z_hash[self.root_index]) if self.root_index is not None else 0

    def _advance_root(self, new_root_index: int) -> None:
        self.root_index = new_root_index

    def _reset_tree(self, root_hash: int, player_piece: Optional[int]) -> None:
        piece = player_piece if player_piece is not None else (int(self.flat_tree.player_piece[self.root_index]) if self.root_index is not None else 1)
        self.root_index = self.flat_tree.create_root(root_hash, piece)
