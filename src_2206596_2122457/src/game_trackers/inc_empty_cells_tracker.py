import numpy as np
from src_2206596_2122457.src.util.board_utils import action_to_id
from src_2206596_2122457.src.constants import BOARD_SIZE
from seahorse.game.game_layout.board import Piece

class IncrementalEmptyCellsTracker:
    __slots__ = ("board_size", "available_actions", "action_id_to_index", "mask", "_count")

    def __init__(self):
        self.board_size: int = BOARD_SIZE
        self.available_actions: np.ndarray = np.empty(self.board_size * self.board_size, dtype=np.int32)
        self.action_id_to_index: np.ndarray = np.full(self.board_size * self.board_size, -1, dtype=np.int32)
        self.mask: np.ndarray = np.zeros(self.board_size * self.board_size, dtype=np.bool_)
        self._count: int = 0

    def reset_from_env(self, env: dict[tuple[int, int], Piece]) -> None:
        board_size = self.board_size
        to_id = action_to_id
        self.mask.fill(False)
        self.action_id_to_index.fill(-1)
        count = 0
        for i in range(board_size):
            for j in range(board_size):
                cell = env.get((i, j), -1)
                if cell == -1:
                    aid = to_id((i, j))
                    self.available_actions[count] = aid
                    self.action_id_to_index[aid] = count
                    self.mask[aid] = True
                    count += 1
        self._count = count

    def remove(self, action_id: int) -> None:
        idx = self.action_id_to_index[action_id]
        if idx == -1:
            return
        last_idx = self._count - 1
        last_action_id = self.available_actions[last_idx]
        if idx != last_idx:
            self.available_actions[idx] = last_action_id
            self.action_id_to_index[last_action_id] = idx
        self.action_id_to_index[action_id] = -1
        self.mask[action_id] = False
        self._count -= 1

    def add(self, action_id: int) -> None:
        if self.mask[action_id]:
            return
        idx = self._count
        self.available_actions[idx] = action_id
        self.action_id_to_index[action_id] = idx
        self.mask[action_id] = True
        self._count += 1

    def possible_actions(self) -> np.ndarray:
        return self.available_actions[:self._count]

    def get_empty_count(self) -> int:
        return self._count

    def is_empty(self, action_id: int) -> bool:
        return self.mask[action_id]
