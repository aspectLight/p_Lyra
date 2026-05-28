from typing import Dict, List, Optional, Tuple, Mapping
import numpy as np
from numba import njit, int32, int8, uint64
from src_2206596_2122457.src.cluster_system.clusters import Clusters
from src_2206596_2122457.src.precomputed.local_pattern_table import (
    PROPAGATION_RADIUS,
    INDEX_TO_CELL,
    CELLS_AFFECTED_BY,
    INITIAL_BOARD_STATE,
    OFFSET_LOOKUP_MIN_D,
    OFFSET_LIST,
    OFFSET_TABLE,
    PATTERN_TABLE,
    SEED,
)
from src_2206596_2122457.src.precomputed.cell_to_index_lookup import CELL_TO_INDEX_LOOKUP
from src_2206596_2122457.src.constants import (
    PIECE_EMPTY,
    PIECE_B,
    PIECE_R,
    PIECE_B_EDGE,
    PIECE_R_EDGE,
    BOARD_SIZE
)
from src_2206596_2122457.src.precomputed.neighbor_offset_tables import NEIGHBOR_TABLE, AFFECTED_REVERSE_TABLE


# -------------------------------
# 🔹 Numba-accelerated helpers
# -------------------------------

@njit(uint64(int32, int16[:, :], int8[:], uint64[:, :]))
def compute_pattern_key_numba(center_idx, neighbor_table, piece_types, xor_masks):
    """Fast C-level XOR accumulation of pattern key."""
    key = np.uint64(0)
    n_offsets = neighbor_table.shape[1]
    for offset_idx in range(n_offsets):
        n_idx = neighbor_table[center_idx, offset_idx]
        if n_idx >= 0:
            p = piece_types[n_idx]
            key ^= xor_masks[p, offset_idx]
    return key


@njit
def refresh_piece_type_cache_numba(board_state, clusters_cell_to_cluster,
                                   clusters_touch_flags, cluster_version,
                                   piece_type_cache, cache_version,
                                   piece_b, piece_r, piece_b_edge, piece_r_edge):
    """Recompute piece type cache in bulk."""
    n = len(board_state)
    for i in range(n):
        piece = board_state[i]
        if piece == piece_b or piece == piece_r:
            cluster_id = clusters_cell_to_cluster[i]
            out_type = piece
            if cluster_id >= 0:
                flags = clusters_touch_flags[cluster_id]
                if piece == piece_b and (flags[0] or flags[1]):
                    out_type = piece_b_edge
                elif piece == piece_r and (flags[2] or flags[3]):
                    out_type = piece_r_edge
            piece_type_cache[i] = out_type
        else:
            piece_type_cache[i] = piece
        cache_version[i] = cluster_version


# -------------------------------
# 🔹 Optimized main class
# -------------------------------

class FieldPropagationPatternStore:
    """
    Optimized version of FieldPropagationPatternStore.
    Uses Numba JIT for pattern key and cache computations.
    """

    __slots__ = (
        'board_size', 'propagation_radius', 'seed',
        'index_to_cell', 'cell_to_index_array', 'cells_affected_by',
        'offset_list', 'offset_table', 'pattern_table', 'min_d',
        'board_state', 'clusters', 'neighbor_table', 'affected_reverse_table',
        'offset_index', '_piece_type_cache', '_piece_type_cache_version',
        '_cluster_version', 'xor_masks'
    )

    def __init__(self, clusters: Optional[Clusters] = None) -> None:
        self.board_size: int = BOARD_SIZE
        self.propagation_radius: int = PROPAGATION_RADIUS
        self.seed: int = SEED

        self.index_to_cell: np.ndarray = np.ascontiguousarray(INDEX_TO_CELL)
        self.cell_to_index_array: np.ndarray = np.ascontiguousarray(CELL_TO_INDEX_LOOKUP, dtype=np.int16)
        self.cells_affected_by: Dict[int, List[int]] = CELLS_AFFECTED_BY
        self.offset_list: List[Tuple[int, int]] = OFFSET_LIST
        self.offset_table: Dict[int, List[int]] = OFFSET_TABLE
        self.pattern_table: Mapping[int, Mapping[int, Mapping[str, float]]] = PATTERN_TABLE
        self.min_d: int = OFFSET_LOOKUP_MIN_D

        n_cells = len(INITIAL_BOARD_STATE)
        self.board_state: np.ndarray = np.ascontiguousarray(
            np.full(n_cells, PIECE_EMPTY, dtype=np.int8)
        )

        self.clusters: Optional[Clusters] = clusters
        self.neighbor_table = np.ascontiguousarray(NEIGHBOR_TABLE, dtype=np.int16)
        self.affected_reverse_table = np.array(AFFECTED_REVERSE_TABLE, dtype=object)

        self.offset_index: Dict[Tuple[int, int], int] = {
            offset: idx for idx, offset in enumerate(self.offset_list)
        }

        # Caches
        self._piece_type_cache: np.ndarray = np.full(n_cells, PIECE_EMPTY, dtype=np.int8)
        self._piece_type_cache_version: np.ndarray = np.full(n_cells, -1, dtype=np.int32)
        self._cluster_version: int = 0

        # Precompute XOR masks
        max_piece_code = max(self.offset_table.keys())
        self.xor_masks = np.ascontiguousarray([
            np.array(self.offset_table.get(p, [0] * len(self.offset_list)), dtype=np.uint64)
            for p in range(max_piece_code + 1)
        ], dtype=np.uint64)

    # -------------------------------
    # Core optimized operations
    # -------------------------------

    def refresh_piece_types(self):
        """Refresh piece-type cache in one Numba call."""
        if self.clusters is None:
            self._piece_type_cache[:] = self.board_state
            self._piece_type_cache_version[:] = self._cluster_version
            return

        refresh_piece_type_cache_numba(
            self.board_state,
            self.clusters.cell_to_cluster,
            self.clusters.cluster_touch_flags_array,
            self._cluster_version,
            self._piece_type_cache,
            self._piece_type_cache_version,
            PIECE_B, PIECE_R, PIECE_B_EDGE, PIECE_R_EDGE
        )

    def _compute_pattern_key_xor(self, center_q: int, center_r: int, player: int) -> int:
        center_idx = self._get_cell_index(center_q, center_r)
        return int(
            compute_pattern_key_numba(center_idx, self.neighbor_table,
                                      self._piece_type_cache, self.xor_masks)
        )

    def get_pattern_key(self, q: int, r: int, player: int) -> int:
        return self._compute_pattern_key_xor(q, r, player)

    def _match_pattern(self, center_q: int, center_r: int, player: int) -> Mapping[str, float]:
        key = self._compute_pattern_key_xor(center_q, center_r, player)
        return self.pattern_table[player][key]

    # -------------------------------
    # Housekeeping methods
    # -------------------------------

    def _get_cell_index(self, q: int, r: int) -> int:
        if q < 0 or r < 0 or q >= self.cell_to_index_array.shape[0] or r >= self.cell_to_index_array.shape[1]:
            raise IndexError(f"Invalid cell coords: {(q, r)}")
        idx: int = int(self.cell_to_index_array[q, r])
        if idx < 0:
            raise IndexError(f"Invalid cell index at: {(q, r)}")
        return idx

    def get_affected_reverse(self, cell_idx: int) -> List[Tuple[int, int]]:
        return self.affected_reverse_table[cell_idx]

    def reset(self) -> None:
        self.board_state.fill(PIECE_EMPTY)
        self._piece_type_cache.fill(PIECE_EMPTY)
        self._piece_type_cache_version.fill(-1)

    def apply_move(self, q: int, r: int, piece_code: int) -> None:
        cell_idx: int = self._get_cell_index(q, r)
        self.board_state[cell_idx] = piece_code
        self._piece_type_cache_version[cell_idx] = -1

    def initialize_from_board(self, board_array: np.ndarray) -> None:
        self.reset()
        self.board_state[:] = board_array
        self.refresh_piece_types()

    def update_clusters(self, clusters: Optional[Clusters]) -> None:
        self.clusters = clusters
        self._cluster_version += 1
        self._piece_type_cache_version.fill(-1)
        self.refresh_piece_types()
