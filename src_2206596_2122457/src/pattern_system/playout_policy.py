from typing import Optional, Dict

from src_2206596_2122457.src.pattern_system.field_propagation_pattern_store import FieldPropagationPatternStore
from src_2206596_2122457.src.precomputed.global_pattern_table import PATTERN_TABLE as GLOBAL_PATTERN_TABLE # type: ignore
from src_2206596_2122457.src.precomputed.local_pattern_table import PATTERN_TABLE as LOCAL_PATTERN_TABLE # type: ignore
from src_2206596_2122457.src.constants import PIECE_B, PIECE_R, BOARD_SIZE, TOTAL_CELLS

import numpy as np

class PlayoutPolicy:
    def __init__(self, store: FieldPropagationPatternStore) -> None:
        self.board_size = BOARD_SIZE
        self.board_size_sq = TOTAL_CELLS

        self.position_coords: np.ndarray = np.array(
            [(i // self.board_size, i % self.board_size) for i in range(self.board_size_sq)],
            dtype=np.int32
        )

        self.global_weights_B: np.ndarray = np.ones(self.board_size_sq, dtype=np.float32)
        self.global_weights_R: np.ndarray = np.ones(self.board_size_sq, dtype=np.float32)
        
        self.active_mask_B: np.ndarray = np.zeros(self.board_size_sq, dtype=bool)
        self.active_mask_R: np.ndarray = np.zeros(self.board_size_sq, dtype=bool)
        self.active_positions_B: np.ndarray = np.empty(self.board_size_sq, dtype=np.int32)
        self.active_positions_R: np.ndarray = np.empty(self.board_size_sq, dtype=np.int32)
        self.active_count_B: int = 0
        self.active_count_R: int = 0
        self.active_index_of_B: np.ndarray = np.full(self.board_size_sq, -1, dtype=np.int32)
        self.active_index_of_R: np.ndarray = np.full(self.board_size_sq, -1, dtype=np.int32)
        
        self.global_total_B: np.float32 = np.float32(0.0)
        self.global_total_R: np.float32 = np.float32(0.0)

        self.tmp_positions: np.ndarray = np.empty(self.board_size_sq, dtype=np.int32)
        self.tmp_weights: np.ndarray = np.empty(self.board_size_sq, dtype=np.float32)
        self.tmp_count: int = 0
        
        self.tmp_cumulative: np.ndarray = np.empty(self.board_size_sq, dtype=np.float32)

        self.local_positions: np.ndarray = np.empty(self.board_size_sq, dtype=np.int32)
        self.local_weights: np.ndarray = np.empty(self.board_size_sq, dtype=np.float32)
        self.local_count: int = 0

        self.global_pattern_table: Dict[int, Dict[int, Dict[str, float]]] = GLOBAL_PATTERN_TABLE
        self.global_pattern_table_B: Dict[int, Dict[str, float]] = self.global_pattern_table.get(PIECE_B, {}) # type: ignore
        self.global_pattern_table_R: Dict[int, Dict[str, float]] = self.global_pattern_table.get(PIECE_R, {}) # type: ignore

        self.local_pattern_table: Dict[int, Dict[int, Dict[str, float]]] = LOCAL_PATTERN_TABLE

        self.rng: np.random.Generator = np.random.default_rng()
        self.store: FieldPropagationPatternStore = store
        self.store.initialize_all_pattern_keys()


    def reset(self, empty_cells: np.ndarray) -> None:
        self._initialize_global_weights(empty_cells)

    def initialize_for_playout(self, empty_cells: np.ndarray) -> None:
        self._initialize_global_weights(empty_cells)

    def _initialize_global_weights(self, empty_cells: np.ndarray) -> None:
        self.global_weights_B[:] = 1.0
        self.global_weights_R[:] = 1.0
        self.active_mask_B[:] = False
        self.active_mask_R[:] = False

        positions = empty_cells.astype(np.int32)

        self.active_mask_B[positions] = True
        self.active_mask_R[positions] = True
        self.active_positions_B[:positions.size] = positions
        self.active_positions_R[:positions.size] = positions
        self.active_count_B = int(positions.size)
        self.active_count_R = int(positions.size)
        self.active_index_of_B.fill(-1)
        self.active_index_of_R.fill(-1)
        self.active_index_of_B[positions] = np.arange(positions.size, dtype=np.int32)
        self.active_index_of_R[positions] = np.arange(positions.size, dtype=np.int32)
        
        total_B: float = 0.0
        total_R: float = 0.0
        table_B = self.global_pattern_table_B
        table_R = self.global_pattern_table_R
        for pos in positions:
            key6, key12 = self.store.get_keys(int(pos))
            gamma_B: float = 1.0
            pd_B = table_B.get(int(key12))
            if pd_B is None:
                pd_B = table_B.get(int(key6))
            if pd_B is not None:
                gamma_B = float(pd_B.get("gamma", 1.0))
            self.global_weights_B[int(pos)] = np.float32(gamma_B)
            total_B += gamma_B
            
            gamma_R: float = 1.0
            pd_R = table_R.get(int(key12))
            if pd_R is None:
                pd_R = table_R.get(int(key6))
            if pd_R is not None:
                gamma_R = float(pd_R.get("gamma", 1.0))
            self.global_weights_R[int(pos)] = np.float32(gamma_R)
            total_R += gamma_R
        self.global_total_B = np.float32(total_B)
        self.global_total_R = np.float32(total_R)

    def apply_move(self, move: int, player: int) -> None:
        if player == PIECE_B:
            if self.active_mask_B[move]:
                self.global_total_B -= self.global_weights_B[move]
                self.active_mask_B[move] = False
                count = self.active_count_B
                if count > 0:
                    arr = self.active_positions_B
                    idx_map = self.active_index_of_B
                    idx = int(idx_map[move])
                    last_idx = count - 1
                    last_pos = int(arr[last_idx])
                    arr[idx], arr[last_idx] = arr[last_idx], arr[idx]
                    idx_map[last_pos] = idx
                    idx_map[move] = -1
                    self.active_count_B = last_idx
        else:
            if self.active_mask_R[move]:
                self.global_total_R -= self.global_weights_R[move]
                self.active_mask_R[move] = False
                count = self.active_count_R
                if count > 0:
                    arr = self.active_positions_R
                    idx_map = self.active_index_of_R
                    idx = int(idx_map[move])
                    last_idx = count - 1
                    last_pos = int(arr[last_idx])
                    arr[idx], arr[last_idx] = arr[last_idx], arr[idx]
                    idx_map[last_pos] = idx
                    idx_map[move] = -1
                    self.active_count_R = last_idx

        row, col = self.position_coords[move]
        self.store.apply_move(int(row), int(col), player)

    def generate_move(self, player: int, last_move: Optional[int]) -> Optional[int]:
        self._compute_local_moves(player, last_move)

        local_pos = self.local_positions[:self.local_count]
        local_wt = self.local_weights[:self.local_count]
        local_total = float(np.sum(local_wt[:self.local_count], dtype=np.float32)) if self.local_count > 0 else 0.0

        if player == PIECE_B:
            active_pos = self.active_positions_B[: self.active_count_B]
            player_weights = self.global_weights_B
            global_total = float(self.global_total_B)
        else:
            active_pos = self.active_positions_R[: self.active_count_R]
            player_weights = self.global_weights_R
            global_total = float(self.global_total_R)

        if self.local_count == 0 and active_pos.size == 0:
            return None

        grand_total = local_total + global_total
        if grand_total <= 0.0:
            return None

        r = float(self.rng.random()) * grand_total
        if r < local_total and self.local_count > 0:
            np.divide(local_wt[:self.local_count], local_total, out=self.tmp_weights[:self.local_count])
            np.cumsum(self.tmp_weights[:self.local_count], out=self.tmp_cumulative[:self.local_count])
            u = float(self.rng.random())
            idx = int(np.searchsorted(self.tmp_cumulative[:self.local_count], u))
            idx = min(idx, self.local_count - 1)
            return int(local_pos[idx])
        else:
            if active_pos.size == 0 or global_total <= 0.0:
                return None
            gw = player_weights[active_pos]
            np.divide(gw, global_total, out=self.tmp_weights[:gw.size])
            np.cumsum(self.tmp_weights[:gw.size], out=self.tmp_cumulative[:gw.size])
            u = float(self.rng.random())
            idx = int(np.searchsorted(self.tmp_cumulative[:gw.size], u))
            idx = min(idx, gw.size - 1)
            return int(active_pos[idx])

    def _compute_local_moves(self, player: int, last_move: Optional[int]) -> None:
        self.local_count = 0

        if last_move is None:
            return

        store: FieldPropagationPatternStore = self.store
        cells_affected: np.ndarray = store.get_affected_neighbors(int(last_move))
        local_pos: np.ndarray = self.local_positions
        local_wt: np.ndarray = self.local_weights
        active_mask: np.ndarray = self.active_mask_B if player == PIECE_B else self.active_mask_R
        get_keys = store.get_keys
        local_table_player: Dict[int, Dict[str, float]] = self.local_pattern_table.get(player, {})

        for affected_idx in cells_affected:
            ai: int = int(affected_idx)
            if not bool(active_mask[ai]):
                continue
            key6, key12 = get_keys(ai)
            pd = local_table_player.get(int(key12))
            if pd is None:
                pd = local_table_player.get(int(key6))
            if pd is None:
                continue
            gamma = float(pd.get("gamma", 0.0))
            if gamma > 0.0:
                local_pos[self.local_count] = ai
                local_wt[self.local_count] = gamma
                self.local_count += 1