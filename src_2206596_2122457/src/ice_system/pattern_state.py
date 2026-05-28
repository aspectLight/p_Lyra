import numpy as np
from typing import Tuple, List

from src_2206596_2122457.pipeline.scripts.ring_constants import BITS_PER_SLICE
from src_2206596_2122457.src.ice_system.ice_constants import (
    MATCH_MODE_ALL, MATCH_MODE_FIRST, PATTERN_PRIMARY_MOVE_RADIUS, PATTERN_NUM_SLICES,
    SLICE_FEATURE_CELLS, SLICE_FEATURE_RED_STONES,
    SLICE_FEATURE_BLUE_STONES, SLICE_FEATURE_PRIMARY_MARKED,
    SLICE_FEATURE_SECONDARY_MARKED, FLAG_HAS_EMPTY_CELLS, FLAG_HAS_PRIMARY_MOVES,
    FLAG_HAS_SECONDARY_MOVES, CELL_EMPTY, CELL_BLUE, CELL_RED, MAX_SLICE_BITS
)
from src_2206596_2122457.src.ice_system.ice_pattern import IcePattern, RotatedPattern
from src_2206596_2122457.src.ice_system.ring_godel import RingGodel
from src_2206596_2122457.src.ice_system.hashed_pattern_set import HashedPatternSet
from src_2206596_2122457.src.precomputed.precomputed_matcher_tables import (
    TABLE_SLICE_INDEX, TABLE_GODEL_BITMASK, TABLE_EDGE_GODEL_BITMASK,
    TABLE_INVERSE_CELL_LOOKUP, IS_EDGE_CELL, EDGE_INDEX_OF_CELL
)
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP 

# Mask to keep values within uint32 bounds
UINT32_MASK = np.uint32(0xFFFFFFFF)


class PatternHit:
    __slots__ = ('pattern_id', 'empty_cells', 'primary_moves', 'secondary_moves')
    
    def __init__(
        self,
        pattern_id: int,
        empty_cells: np.ndarray,
        primary_moves: np.ndarray,
        secondary_moves: np.ndarray | None = None
    ) -> None:
        self.pattern_id = pattern_id
        self.empty_cells = empty_cells
        self.primary_moves = primary_moves
        self.secondary_moves = secondary_moves if secondary_moves is not None else np.array([], dtype=np.int32)


class PatternMatcherData:
    __slots__ = (
        'num_cells', 'is_edge_cell', 'edge_index_of_cell',
        'table_slice_index', 'table_godel_bitmask', 'table_edge_godel_bitmask',
        'table_inverse_cell_lookup'
    )
    
    def __init__(self) -> None:
        self.table_slice_index = TABLE_SLICE_INDEX
        self.table_godel_bitmask = TABLE_GODEL_BITMASK
        self.table_edge_godel_bitmask = TABLE_EDGE_GODEL_BITMASK
        self.table_inverse_cell_lookup = TABLE_INVERSE_CELL_LOOKUP
        self.is_edge_cell = IS_EDGE_CELL
        self.edge_index_of_cell = EDGE_INDEX_OF_CELL
        self.num_cells = len(self.is_edge_cell)
    
    def get_rotated_move(self, center_cell_idx: int, slice_idx: int, bit_position: int, rotation_angle: int) -> int:
        rotated_slice_idx = (slice_idx + PATTERN_NUM_SLICES - rotation_angle) % PATTERN_NUM_SLICES
        return int(self.table_inverse_cell_lookup[center_cell_idx, rotated_slice_idx, bit_position])


class PatternState:
    __slots__ = (
        'board_state', 'empty_cell_positions', 'matcher_data',
        'update_radius', 'slice_godel_values', 'ring_godel_values'
    )
    
    def __init__(
        self,
        board_state: np.ndarray,
        matcher_data: PatternMatcherData | None = None
    ) -> None:
        self.board_state = np.asarray(board_state, dtype=np.int32)
        self.matcher_data = matcher_data if matcher_data is not None else PatternMatcherData()
        self.update_radius = PATTERN_PRIMARY_MOVE_RADIUS
        
        num_cells = len(self.board_state)
        self.slice_godel_values = np.zeros((num_cells, 2, PATTERN_NUM_SLICES), dtype=np.uint32)
        self.ring_godel_values = np.zeros((num_cells, PATTERN_NUM_SLICES), dtype=np.uint8)
        self.empty_cell_positions = np.nonzero(self.board_state == CELL_EMPTY)[0]
    
    def clear_godel_values(self) -> None:
        self.slice_godel_values.fill(0)
        self.ring_godel_values.fill(0)
    
    def copy_state(self, other_state: "PatternState") -> None:
        self.board_state[:] = other_state.board_state
        self.slice_godel_values[:] = other_state.slice_godel_values
        self.ring_godel_values[:] = other_state.ring_godel_values
        self.empty_cell_positions = other_state.empty_cell_positions
    
    def update_ring_godel(self, center_cell_idx: int) -> None:
        self.ring_godel_values[center_cell_idx].fill(0)
        
        for slice_idx in range(PATTERN_NUM_SLICES):
            blue_or_red_godel = (self.slice_godel_values[center_cell_idx, 0, slice_idx] | 
                                self.slice_godel_values[center_cell_idx, 1, slice_idx])
            if blue_or_red_godel & 1:
                if self.slice_godel_values[center_cell_idx, 0, slice_idx] & 1:
                    self.ring_godel_values[center_cell_idx, slice_idx] |= 1
                if self.slice_godel_values[center_cell_idx, 1, slice_idx] & 1:
                    self.ring_godel_values[center_cell_idx, slice_idx] |= 2
    
    def update_cell(self, cell_idx: int) -> None:
        if not (0 <= cell_idx < len(self.board_state)):
            return
        
        cell_color = self.board_state[cell_idx]
        neighbor_cells = NEIGHBORS_LOOKUP[cell_idx]
        
        for observer_cell_idx_raw in neighbor_cells:
            observer_cell_idx = int(observer_cell_idx_raw)
            if observer_cell_idx == -1:
                continue
            
            slice_idx_raw = self.matcher_data.table_slice_index[observer_cell_idx, cell_idx]
            slice_idx = int(slice_idx_raw)
            if slice_idx == -1:
                continue
            
            godel_cell_mask = np.uint32(self.matcher_data.table_godel_bitmask[observer_cell_idx, cell_idx])
            
            if cell_color == CELL_BLUE:
                self.slice_godel_values[observer_cell_idx, 0, slice_idx] |= godel_cell_mask
                if godel_cell_mask & 1:
                    self.ring_godel_values[observer_cell_idx, slice_idx] |= 1
            
            elif cell_color == CELL_RED:
                self.slice_godel_values[observer_cell_idx, 1, slice_idx] |= godel_cell_mask
                if godel_cell_mask & 1:
                    self.ring_godel_values[observer_cell_idx, slice_idx] |= 2
            
            elif cell_color == CELL_EMPTY:
                # Use XOR with mask to avoid negative number overflow
                self.slice_godel_values[observer_cell_idx, 0, slice_idx] &= UINT32_MASK ^ godel_cell_mask
                self.slice_godel_values[observer_cell_idx, 1, slice_idx] &= UINT32_MASK ^ godel_cell_mask
                if godel_cell_mask & 1:
                    self.ring_godel_values[observer_cell_idx, slice_idx] = 0
        
        if self.matcher_data.is_edge_cell[cell_idx]:
            edge_idx = int(self.matcher_data.edge_index_of_cell[cell_idx])
            for observer_cell_idx_raw in neighbor_cells:
                observer_cell_idx = int(observer_cell_idx_raw)
                if observer_cell_idx == -1:
                    continue
                
                for slice_idx in range(PATTERN_NUM_SLICES):
                    edge_godel_cell_mask = np.uint32(self.matcher_data.table_edge_godel_bitmask[observer_cell_idx, edge_idx, slice_idx])
                    
                    if cell_color == CELL_BLUE:
                        self.slice_godel_values[observer_cell_idx, 0, slice_idx] |= edge_godel_cell_mask
                    elif cell_color == CELL_RED:
                        self.slice_godel_values[observer_cell_idx, 1, slice_idx] |= edge_godel_cell_mask
                    elif cell_color == CELL_EMPTY:
                        # Use XOR with mask to avoid negative number overflow
                        self.slice_godel_values[observer_cell_idx, 0, slice_idx] &= UINT32_MASK ^ edge_godel_cell_mask
                        self.slice_godel_values[observer_cell_idx, 1, slice_idx] &= UINT32_MASK ^ edge_godel_cell_mask
    
    def update_changed_cells(self, changed_cell_indices: np.ndarray) -> None:
        for cell_idx in changed_cell_indices:
            self.update_cell(int(cell_idx))
    
    def update_all_cells(self) -> None:
        self.clear_godel_values()
        occupied_cell_indices = np.nonzero(self.board_state != CELL_EMPTY)[0]
        for cell_idx in occupied_cell_indices:
            self.update_cell(int(cell_idx))
    
    def check_rotated_slices(self, center_cell_idx: int, pattern: IcePattern, rotation_angle: int) -> bool:
        blue_stone_masks = self.slice_godel_values[center_cell_idx, 0]
        red_stone_masks = self.slice_godel_values[center_cell_idx, 1]
        pattern_slices = pattern.slices
        
        for slice_offset in range(PATTERN_NUM_SLICES):
            rotated_slice_idx = (rotation_angle + slice_offset) % PATTERN_NUM_SLICES
            
            pattern_cell_bitmask = int(pattern_slices[rotated_slice_idx, SLICE_FEATURE_CELLS])
            blue_stones_present = int(blue_stone_masks[slice_offset]) & pattern_cell_bitmask
            red_stones_present = int(red_stone_masks[slice_offset]) & pattern_cell_bitmask
            occupied_cells = blue_stones_present | red_stones_present
            
            pattern_blue_stones = int(pattern_slices[rotated_slice_idx, SLICE_FEATURE_BLUE_STONES])
            pattern_red_stones = int(pattern_slices[rotated_slice_idx, SLICE_FEATURE_RED_STONES])
            pattern_empty_cells = pattern_cell_bitmask - pattern_blue_stones - pattern_red_stones
            
            if (occupied_cells & pattern_empty_cells) != 0:
                return False
            if (blue_stones_present & pattern_blue_stones) != pattern_blue_stones:
                return False
            if (red_stones_present & pattern_red_stones) != pattern_red_stones:
                return False
        
        return True
    
    def check_ring_godel(self, center_cell_idx: int, pattern: IcePattern, rotation_angle: int) -> bool:
        pattern_slices = pattern.slices
        
        for slice_idx in range(PATTERN_NUM_SLICES):
            rotated_slice_idx = (rotation_angle + slice_idx) % PATTERN_NUM_SLICES
            
            pattern_ring_requirement = int(pattern_slices[rotated_slice_idx, SLICE_FEATURE_CELLS]) & 3
            cell_ring_value = int(self.ring_godel_values[center_cell_idx, slice_idx])
            
            if (pattern_ring_requirement & 1) and not (cell_ring_value & 1):
                return False
            if (pattern_ring_requirement & 2) and not (cell_ring_value & 2):
                return False
        
        return True
    
    def check_rotated_pattern(
        self,
        center_cell_idx: int,
        rotated_pattern: RotatedPattern
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        pattern: IcePattern = rotated_pattern.pattern
        
        if not self.check_ring_godel(center_cell_idx, pattern, rotated_pattern.angle):
            return (False, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        
        if pattern.radius > 1:
            if not self.check_rotated_slices(center_cell_idx, pattern, rotated_pattern.angle):
                return (False, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        
        matched_empty_cells: np.ndarray = np.array([], dtype=np.int32)
        matched_primary_moves: np.ndarray = np.array([], dtype=np.int32)
        matched_secondary_moves: np.ndarray = np.array([], dtype=np.int32)
        
        pattern_slices = pattern.slices
        
        if pattern.flags & FLAG_HAS_EMPTY_CELLS:
            empty_cell_list: List[int] = []
            for slice_idx in range(PATTERN_NUM_SLICES):
                cell_bitmask: int = int(pattern_slices[slice_idx, SLICE_FEATURE_CELLS])
                blue_stone_bitmask: int = int(pattern_slices[slice_idx, SLICE_FEATURE_BLUE_STONES])
                red_stone_bitmask: int = int(pattern_slices[slice_idx, SLICE_FEATURE_RED_STONES])
                empty_cell_bitmask: int = cell_bitmask & ~(blue_stone_bitmask | red_stone_bitmask)
                
                for bit_pos in range(MAX_SLICE_BITS):
                    if empty_cell_bitmask & (1 << bit_pos):
                        matched_cell_idx: int = self.matcher_data.get_rotated_move(center_cell_idx, slice_idx, bit_pos, rotated_pattern.angle)
                        if matched_cell_idx != -1:
                            empty_cell_list.append(matched_cell_idx)
            
            matched_empty_cells = np.asarray(empty_cell_list, dtype=np.int32)
        
        if pattern.flags & FLAG_HAS_PRIMARY_MOVES:
            primary_move_list: List[int] = []
            for slice_idx in range(PATTERN_NUM_SLICES):
                primary_marked_bitmask: int = int(pattern_slices[slice_idx, SLICE_FEATURE_PRIMARY_MARKED])
                
                for bit_pos in range(MAX_SLICE_BITS):
                    if primary_marked_bitmask & (1 << bit_pos):
                        matched_cell_idx: int = self.matcher_data.get_rotated_move(center_cell_idx, slice_idx, bit_pos, rotated_pattern.angle)
                        if matched_cell_idx != -1:
                            primary_move_list.append(matched_cell_idx)
            
            matched_primary_moves = np.asarray(primary_move_list, dtype=np.int32)
        
        if pattern.flags & FLAG_HAS_SECONDARY_MOVES:
            secondary_move_list: List[int] = []
            for slice_idx in range(PATTERN_NUM_SLICES):
                secondary_marked_bitmask: int = int(pattern_slices[slice_idx, SLICE_FEATURE_SECONDARY_MARKED])
                
                for bit_pos in range(MAX_SLICE_BITS):
                    if secondary_marked_bitmask & (1 << bit_pos):
                        matched_cell_idx: int = self.matcher_data.get_rotated_move(center_cell_idx, slice_idx, bit_pos, rotated_pattern.angle)
                        if matched_cell_idx != -1:
                            secondary_move_list.append(matched_cell_idx)
            
            matched_secondary_moves = np.asarray(secondary_move_list, dtype=np.int32)
        
        return (True, matched_empty_cells, matched_primary_moves, matched_secondary_moves)
    
    def match_on_cell(self, pattern_set: HashedPatternSet, center_cell_idx: int, match_mode: int = MATCH_MODE_ALL) -> np.ndarray:
        ring_godel_value = 0
        for slice_idx in range(PATTERN_NUM_SLICES):
            slice_value = int(self.ring_godel_values[center_cell_idx, slice_idx])
            shift = slice_idx * BITS_PER_SLICE
            ring_godel_value |= (slice_value << shift)
        
        ring_godel_obj = RingGodel(ring_godel_value)
        rotated_patterns_list: np.ndarray = pattern_set.list_for_godel(ring_godel_obj)
        
        pattern_hits_list: List[PatternHit] = []
        
        for rotated_pattern_obj in rotated_patterns_list:
            rotated_pattern: RotatedPattern = rotated_pattern_obj
            is_matched, matched_empty_cells, matched_primary_moves, matched_secondary_moves = self.check_rotated_pattern(center_cell_idx, rotated_pattern)
            
            if is_matched:
                matched_pattern_id: int = rotated_pattern.pattern.index
                pattern_hit_obj: PatternHit = PatternHit(matched_pattern_id, matched_empty_cells, matched_primary_moves, matched_secondary_moves)
                pattern_hits_list.append(pattern_hit_obj)
                
                if match_mode == MATCH_MODE_FIRST:
                    break
        
        pattern_hits = np.asarray(pattern_hits_list, dtype=object)
        return pattern_hits
    
    def match_on_board(
        self,
        consider_cells: np.ndarray,
        pattern_set: HashedPatternSet,
        match_mode: int = MATCH_MODE_ALL
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_cells = len(self.board_state)
        
        if consider_cells.ndim == 1 and consider_cells.dtype in [np.int32, np.int64]:
            cell_indices_to_check = consider_cells
        else:
            cell_indices_to_check = np.nonzero(consider_cells)[0]
        
        cell_match_mask = np.zeros(num_cells, dtype=bool)
        all_pattern_hits = np.empty(num_cells, dtype=object)
        
        for i in range(num_cells):
            all_pattern_hits[i] = np.array([], dtype=object)
        
        for center_cell_idx in cell_indices_to_check:
            cell_pattern_hits = self.match_on_cell(pattern_set, int(center_cell_idx), match_mode)
            
            if len(cell_pattern_hits) > 0:
                cell_match_mask[center_cell_idx] = True
                all_pattern_hits[center_cell_idx] = cell_pattern_hits
        
        return (cell_match_mask, all_pattern_hits)