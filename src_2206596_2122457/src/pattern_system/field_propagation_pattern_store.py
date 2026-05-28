from typing import Dict, List, Tuple, Mapping
import numpy as np
from src_2206596_2122457.src.precomputed.local_pattern_table import NEIGHBOR_MAP, OFFSET_LIST, OFFSET_LOOKUP_MIN_D, OFFSET_TABLE, PATTERN_TABLE, SEED, INITIAL_BOARD_STATE, INDEX_TO_CELL # type: ignore
from src_2206596_2122457.src.precomputed.cell_to_index_lookup import CELL_TO_INDEX_LOOKUP
from src_2206596_2122457.src.constants import (
    PIECE_EMPTY,
    PIECE_B,
    PIECE_R,
    PIECE_B_EDGE,
    PIECE_R_EDGE,
    BOARD_SIZE,
    TOTAL_CELLS
)
from src_2206596_2122457.src.precomputed.neighbor_offset_tables import NEIGHBOR_TABLE, AFFECTED_REVERSE_TABLE
from src_2206596_2122457.src.precomputed.xor_masks import XOR_MASKS 

class FieldPropagationPatternStore:
    __slots__ = (
        'board_size', 'seed',
        'index_to_cell', 'cell_to_index_array', 'neighbor_map',
        'offset_list', 'offset_table', 'pattern_table', 'min_d',
        'board_state', 'clusters', 'neighbor_table', 'affected_reverse_table',
        'offset_index', 'xor_masks', 'pattern_keys'
    )
    
    def __init__(self) -> None:
        self.board_size: int = BOARD_SIZE
        self.seed: int = SEED
        self.cell_to_index_array: np.ndarray = np.ascontiguousarray(CELL_TO_INDEX_LOOKUP, dtype=np.uint16)
        self.neighbor_map: List[np.ndarray] = NEIGHBOR_MAP
        self.offset_list: List[Tuple[int, int]] = OFFSET_LIST
        self.offset_table: Dict[int, List[int]] = OFFSET_TABLE
        self.index_to_cell: np.ndarray = np.ascontiguousarray(INDEX_TO_CELL)
        self.pattern_table: Mapping[int, Mapping[int, Mapping[str, float]]] = PATTERN_TABLE # type: ignore
        self.min_d: int = OFFSET_LOOKUP_MIN_D
        self.neighbor_table = np.ascontiguousarray(NEIGHBOR_TABLE, dtype=np.int16)
        self.affected_reverse_table = np.array(AFFECTED_REVERSE_TABLE, dtype=object)
        self.xor_masks: np.ndarray = np.ascontiguousarray(XOR_MASKS, dtype=np.uint64)
        self.pattern_keys: np.ndarray = np.zeros((TOTAL_CELLS, 2), dtype=np.uint64)
        self.board_state: np.ndarray = np.ascontiguousarray(np.full(len(INITIAL_BOARD_STATE), PIECE_EMPTY, dtype=np.int8))
        
        self.offset_index: Dict[Tuple[int, int], int] = {
            offset: idx for idx, offset in enumerate(self.offset_list)
        }

    
    def _compute_key_from_scratch(self, center_idx: int) -> Tuple[int, int]:
        neighbor_indices = self.neighbor_table[center_idx]
        types = np.full_like(neighbor_indices, PIECE_EMPTY, dtype=np.int8)
        center_q, center_r = self.index_to_cell[center_idx]
        board_size = self.board_size

        for i in range(len(neighbor_indices)):
            neighbor_idx = int(neighbor_indices[i])
            if neighbor_idx != -1:
                types[i] = int(self.board_state[neighbor_idx])
            else:
                dq, dr = self.offset_list[i + 1]
                qn = int(center_q) + int(dq)
                rn = int(center_r) + int(dr)

                if qn < 0 or qn >= board_size:
                    types[i] = PIECE_B_EDGE   
                elif rn < 0 or rn >= board_size:
                    types[i] = PIECE_R_EDGE
                else:
                    types[i] = PIECE_EMPTY

        idxs = np.arange(1, len(types) + 1)
        key6 = int(np.bitwise_xor.reduce(self.xor_masks[types[:6], idxs[:6]].astype(np.uint64)))
        key12 = int(np.bitwise_xor.reduce(self.xor_masks[types[:12], idxs[:12]].astype(np.uint64)))
        return key6, key12

    def _get_cell_index(self, q: int, r: int) -> int:
        if q < 0 or r < 0 or q >= self.cell_to_index_array.shape[0] or r >= self.cell_to_index_array.shape[1]:
            raise IndexError(f"Invalid cell coords: {(q, r)}")
        idx: int = int(self.cell_to_index_array[q, r])
        if idx < 0:
            raise IndexError(f"Invalid cell index at: {(q, r)}")
        return idx

    def initialize_all_pattern_keys(self) -> None:
        self.pattern_keys.fill(0)
        for cell_idx in range(TOTAL_CELLS):
            if int(self.board_state[cell_idx]) == PIECE_EMPTY:
                key6, key12 = self._compute_key_from_scratch(cell_idx)
                self.pattern_keys[cell_idx, 0] = np.uint64(key6)
                self.pattern_keys[cell_idx, 1] = np.uint64(key12)
            else:
                self.pattern_keys[cell_idx, 0] = np.uint64(0)
                self.pattern_keys[cell_idx, 1] = np.uint64(0)

    def reset(self) -> None:
        self.board_state.fill(PIECE_EMPTY)
        self.pattern_keys.fill(0)

    def apply_move(self, q: int, r: int, piece_code: int) -> None:
        cell_idx: int = self._get_cell_index(q, r)
        self.board_state[cell_idx] = piece_code
        self.pattern_keys[cell_idx, 0] = np.uint64(0)
        self.pattern_keys[cell_idx, 1] = np.uint64(0)

        for neighbor_idx, offset_idx in self.affected_reverse_table[cell_idx]:
            n_idx = int(neighbor_idx)
            if int(self.board_state[n_idx]) != PIECE_EMPTY:
                continue
            oi = int(offset_idx)
            hash_to_remove = self.xor_masks[PIECE_EMPTY, oi + 1]
            hash_to_apply = self.xor_masks[piece_code, oi + 1]
            update_xor = np.uint64(hash_to_remove ^ hash_to_apply)
            if oi < 6:
                self.pattern_keys[n_idx, 0] ^= update_xor
                self.pattern_keys[n_idx, 1] ^= update_xor
            elif oi < 12:
                self.pattern_keys[n_idx, 1] ^= update_xor
            else:
                pass

    def initialize_from_board(self, board_array: np.ndarray) -> None:
        self.reset()
        for idx, piece_code in enumerate(board_array):
            if piece_code in (PIECE_B, PIECE_R):
                self.board_state[idx] = piece_code
        self.initialize_all_pattern_keys()

    def get_keys(self, cell_idx: int) -> Tuple[int, int]:
        return int(self.pattern_keys[cell_idx, 0]), int(self.pattern_keys[cell_idx, 1])

    def get_affected_neighbors(self, last_move: int) -> np.ndarray:
        if 0 <= last_move < len(self.neighbor_map):
            return self.neighbor_map[last_move]
        return np.array([], dtype=np.int32)

    def copy(self) -> "FieldPropagationPatternStore":
        new_store = FieldPropagationPatternStore.__new__(FieldPropagationPatternStore)
        new_store.board_size = self.board_size
        new_store.seed = self.seed
        new_store.index_to_cell = self.index_to_cell
        new_store.cell_to_index_array = self.cell_to_index_array
        new_store.neighbor_map = self.neighbor_map
        new_store.offset_list = self.offset_list
        new_store.offset_table = self.offset_table
        new_store.pattern_table = self.pattern_table
        new_store.min_d = self.min_d
        new_store.neighbor_table = self.neighbor_table
        new_store.affected_reverse_table = self.affected_reverse_table
        new_store.offset_index = self.offset_index
        new_store.xor_masks = self.xor_masks
        new_store.board_state = self.board_state.copy()
        new_store.pattern_keys = self.pattern_keys.copy()
        return new_store