import numpy as np
import numpy.typing as npt

from src_2206596_2122457.src.ice_system.ice_engine.ice_fillin_modes import FillinMode
from src_2206596_2122457.src.group_system.groups import BLUE, BOARD_SIZE
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP


class IceEngineHelpers:
    
    @staticmethod
    def turn_off_capture(mode: FillinMode) -> FillinMode:
        if mode == FillinMode.MONOCOLOR_USING_CAPTURED:
            return FillinMode.MONOCOLOR
        return mode
    
    @staticmethod
    def uses_capture(mode: FillinMode) -> bool:
        return mode in (FillinMode.MONOCOLOR_USING_CAPTURED, FillinMode.BICOLOR)
    
    @staticmethod
    def is_monocolor_using_capture(mode: FillinMode) -> bool:
        return mode == FillinMode.MONOCOLOR_USING_CAPTURED
    
    @staticmethod
    def bicolor(mode: FillinMode) -> bool:
        return mode == FillinMode.BICOLOR
    
    @staticmethod
    def pick_color(
        board_state: npt.NDArray[np.int32],
        color: int,
        p: int,
        mode: FillinMode,
    ) -> int:
        if IceEngineHelpers.bicolor(mode):
            neighbors_raw: npt.NDArray[np.int32] = NEIGHBORS_LOOKUP[p]
            valid_neighbors: npt.NDArray[np.int32] = neighbors_raw[neighbors_raw != -1]
            for n in valid_neighbors:
                n_int: int = int(n)
                if board_state[n_int] != 0:
                    return int(board_state[n_int])
            return color
        return color
    
    @staticmethod
    def add_neighbors_to_consider(
        p: int,
        board_state: npt.NDArray[np.int32],
        consider_mask: npt.NDArray[np.bool_],
    ) -> None:
        neighbors_raw: npt.NDArray[np.int32] = NEIGHBORS_LOOKUP[p]
        for n_raw in neighbors_raw:
            n: int = int(n_raw)
            if n != -1 and board_state[n] == 0:
                consider_mask[n] = True
    
    @staticmethod
    def is_color_edge(cap: int, color: int) -> bool:
        row: int = cap // BOARD_SIZE
        col: int = cap % BOARD_SIZE
        
        if color == BLUE:
            return row == 0 or row == BOARD_SIZE - 1
        else:
            return col == 0 or col == BOARD_SIZE - 1
    
    @staticmethod
    def verify_complete_adjacency(
        cell_indices: npt.NDArray[np.int32],
        exclude: int = -1,
    ) -> bool:
        valid_cells: npt.NDArray[np.int32] = cell_indices[cell_indices != -1]
        if exclude != -1:
            valid_cells = valid_cells[valid_cells != exclude]
        
        if len(valid_cells) < 2:
            return True
        
        for i in range(len(valid_cells)):
            for j in range(i + 1, len(valid_cells)):
                cell_a: int = int(valid_cells[i])
                cell_b: int = int(valid_cells[j])
                
                neighbors_a: npt.NDArray[np.int32] = NEIGHBORS_LOOKUP[cell_a]
                if not np.any(neighbors_a == cell_b):
                    return False
        
        return True