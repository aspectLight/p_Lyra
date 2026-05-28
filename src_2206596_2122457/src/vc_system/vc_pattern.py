import numpy as np
from typing import Tuple, Iterable, Optional
from functools import lru_cache
from numpy.typing import NDArray

from src_2206596_2122457.src.precomputed.precomputed_vc_patterns import (
    NUM_PATTERNS_BLACK,
    NUM_PATTERNS_WHITE,
    PATTERNS_MUST_HAVE_BLACK,
    PATTERNS_NOT_OPPONENT_BLACK,
    PATTERNS_ENDPOINTS_BLACK,
    PATTERNS_MUST_HAVE_WHITE,
    PATTERNS_NOT_OPPONENT_WHITE,
    PATTERNS_ENDPOINTS_WHITE,
)

EAST = 0
WEST = 1
NORTH = 2
SOUTH = 3

class VcPattern:
    __slots__ = ('_must_have', '_not_opponent', '_endpoints', '_width', '_height')
    
    def __init__(
        self,
        table_must_have: NDArray[np.bool_],
        table_not_opponent: NDArray[np.bool_],
        table_endpoints: Tuple[int, int],
        width: int,
        height: int
    ) -> None:
        self._must_have = np.asarray(table_must_have, dtype=np.bool_)
        self._not_opponent = np.asarray(table_not_opponent, dtype=np.bool_)
        self._endpoints = table_endpoints
        self._width = width
        self._height = height
    
    @property
    def must_have(self) -> NDArray[np.bool_]:
        return self._must_have
    
    @property
    def not_opponent(self) -> NDArray[np.bool_]:
        return self._not_opponent
    
    @property
    def endpoints(self) -> Tuple[int, int]:
        return self._endpoints
    
    def matches(self, color: int, board_state: NDArray[np.int_]) -> bool:
        my_color_mask = board_state == color
        opponent_color = 3 - color
        opponent_mask = board_state == opponent_color
        
        not_opponent_condition = not (self._not_opponent & opponent_mask).any()
        must_have_condition = np.all(my_color_mask[self._must_have])
        
        return bool(not_opponent_condition and must_have_condition)
    
    def shifted_copy(
        self,
        direction: int,
        index_map: NDArray[np.int32]
    ) -> Optional['VcPattern']:
        shifted_must = np.zeros_like(self._must_have)
        shifted_not_opp = np.zeros_like(self._not_opponent)
        
        for old_idx in np.where(self._must_have)[0]:
            new_idx = index_map[old_idx]
            if new_idx >= 0:
                shifted_must[new_idx] = True
            else:
                return None
        
        for old_idx in np.where(self._not_opponent)[0]:
            new_idx = index_map[old_idx]
            if new_idx >= 0:
                shifted_not_opp[new_idx] = True
            else:
                return None
        
        ep0_old, ep1_old = self._endpoints
        col0, row0 = self._index_to_coords(ep0_old)
        col1, row1 = self._index_to_coords(ep1_old)
        
        if direction == EAST:
            col0, col1 = col0 + 1, col1 + 1
        elif direction == WEST:
            col0, col1 = col0 - 1, col1 - 1
        elif direction == NORTH:
            row0, row1 = row0 + 1, row1 + 1
        elif direction == SOUTH:
            row0, row1 = row0 - 1, row1 - 1
        
        if (0 <= col0 < self._width and 0 <= row0 < self._height and
            0 <= col1 < self._width and 0 <= row1 < self._height):
            ep0_new = self._coords_to_index(col0, row0)
            ep1_new = self._coords_to_index(col1, row1)
            return VcPattern(
                shifted_must,
                shifted_not_opp,
                (ep0_new, ep1_new),
                self._width,
                self._height
            )
        
        return None
    
    def _coords_to_index(self, col: int, row: int) -> int:
        return row * self._width + col
    
    def _index_to_coords(self, index: int) -> Tuple[int, int]:
        row = index // self._width
        col = index % self._width
        return col, row
    
    def as_packed_bitset(self) -> Tuple[bytes, bytes]:
        packed_must = np.packbits(self._must_have).tobytes()
        packed_not = np.packbits(self._not_opponent).tobytes()
        return packed_must, packed_not


@lru_cache(maxsize=4)
def load_patterns_for_board(width: int, height: int, color: int) -> Tuple[VcPattern, ...]:
    patterns: list[VcPattern] = []
    
    if color == 1:
        num_patterns = int(NUM_PATTERNS_BLACK)
        must_have_array = PATTERNS_MUST_HAVE_BLACK
        not_opponent_array = PATTERNS_NOT_OPPONENT_BLACK
        endpoints_array = PATTERNS_ENDPOINTS_BLACK
    else:
        num_patterns = int(NUM_PATTERNS_WHITE)
        must_have_array = PATTERNS_MUST_HAVE_WHITE
        not_opponent_array = PATTERNS_NOT_OPPONENT_WHITE
        endpoints_array = PATTERNS_ENDPOINTS_WHITE
    
    for i in range(num_patterns):
        must_have = must_have_array[i]
        not_opponent = not_opponent_array[i]
        endpoints = tuple(endpoints_array[i])  # type: ignore
        
        pattern = VcPattern(
            must_have,
            not_opponent,
            endpoints,  # type: ignore
            width,
            height
        )
        patterns.append(pattern)
    
    return tuple(patterns)


class VcPatternManager:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._pattern_cache: dict[int, Tuple[VcPattern, ...]] = {}
    
    def get_patterns_for_color(self, color: int) -> Tuple[VcPattern, ...]:
        if color not in self._pattern_cache:
            self._pattern_cache[color] = load_patterns_for_board(
                self._width,
                self._height,
                color
            )
        return self._pattern_cache[color]
    
    def find_matching_patterns(
        self,
        color: int,
        board_state: NDArray[np.int_]
    ) -> Iterable[VcPattern]:
        patterns = self.get_patterns_for_color(color)
        for pattern in patterns:
            if pattern.matches(color, board_state):
                yield pattern