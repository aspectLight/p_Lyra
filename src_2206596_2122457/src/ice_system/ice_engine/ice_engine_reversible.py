import numpy as np
import numpy.typing as npt

from src_2206596_2122457.src.ice_system.pattern_state import PatternState, PatternHit
from src_2206596_2122457.src.ice_system.ice_pattern_set import IcePatternSet
from src_2206596_2122457.src.group_system.groups import EMPTY, TOTAL_CELLS
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP
from src_2206596_2122457.src.ice_system.ice_constants import MATCH_MODE_FIRST, MATCH_MODE_ALL


class IceEngineReversible:
    
    @staticmethod
    def is_reversible(
        pastate: PatternState,
        color: int,
        p: int,
        patterns: IcePatternSet,
        use_s_reversible_as_reversible: bool,
    ) -> int:
        if p == -1 or p < 0 or p >= TOTAL_CELLS:
            return -1
        
        pastate.update_all_cells()
        
        hits: npt.NDArray[np.object_] = pastate.match_on_cell(
            patterns.hashed_reversible(color), p, MATCH_MODE_FIRST
        )
        
        if len(hits) > 0:
            return int(hits[0].secondary_moves[0])
        
        if not use_s_reversible_as_reversible:
            return -1
        
        hits = pastate.match_on_cell(
            patterns.hashed_s_reversible(color), p, MATCH_MODE_FIRST
        )
        
        if len(hits) > 0:
            return int(hits[0].secondary_moves[0])
        
        board_state: npt.NDArray[np.int32] = pastate.board_state
        occupied: bool = board_state[p] != EMPTY
        
        if occupied:
            board_state[p] = EMPTY
            pastate.update_cell(p)
        
        neighbors_raw: npt.NDArray[np.int32] = NEIGHBORS_LOOKUP[p]
        for n_raw in neighbors_raw:
            q: int = int(n_raw)
            if q == -1:
                continue
            
            if board_state[q] != EMPTY:
                continue
            
            loc_hits: npt.NDArray[np.object_] = pastate.match_on_cell(
                patterns.hashed_s_reversible(color), q, MATCH_MODE_ALL
            )
            
            for hit_obj in loc_hits:
                hit: PatternHit = hit_obj
                others: npt.NDArray[np.int32] = hit.primary_moves
                if np.any(others == p):
                    if occupied:
                        board_state[p] = color
                        pastate.update_cell(p)
                    return int(hit.secondary_moves[0])
        
        if occupied:
            board_state[p] = color
            pastate.update_cell(p)
        
        return -1