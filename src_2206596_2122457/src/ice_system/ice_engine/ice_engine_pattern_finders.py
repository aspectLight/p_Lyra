import numpy as np
import numpy.typing as npt

from src_2206596_2122457.src.ice_system.pattern_state import PatternState, PatternHit
from src_2206596_2122457.src.ice_system.ice_pattern_set import IcePatternSet
from src_2206596_2122457.src.ice_system.inferior_cells import InferiorCells
from src_2206596_2122457.src.group_system.groups import EMPTY, TOTAL_CELLS
from src_2206596_2122457.src.ice_system.ice_constants import MATCH_MODE_FIRST, MATCH_MODE_ALL
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP


class IceEnginePatternFinders:
    
    @staticmethod
    def find_s_reversible(
        pastate: PatternState,
        color: int,
        consider: npt.NDArray[np.bool_],
        inf: InferiorCells,
        patterns: IcePatternSet,
        find_all_killers: bool,
    ) -> None:
        rev_mask: npt.NDArray[np.bool_]
        hits_array: npt.NDArray[np.object_]
        rev_mask, hits_array = pastate.match_on_board(
            consider, patterns.hashed_s_reversible(color), MATCH_MODE_ALL
        )
        
        for p in np.flatnonzero(rev_mask):
            p_int: int = int(p)
            hits: npt.NDArray[np.object_] = hits_array[p_int]
            for hit_obj in hits:
                hit: PatternHit = hit_obj
                empty_positions: npt.NDArray[np.int32] = hit.empty_cells
                if len(empty_positions) == 1:
                    killer: int = int(hit.secondary_moves[0])
                    inf.add_vulnerable(p_int, {killer})
                    if not find_all_killers:
                        break
        
        for p in np.flatnonzero(rev_mask):
            p_int: int = int(p)
            hits = hits_array[p_int]
            for hit_obj in hits:
                hit = hit_obj
                empty_positions = hit.empty_cells
                if len(empty_positions) == 1:
                    continue
                
                others: npt.NDArray[np.int32] = hit.primary_moves
                reverser: int = int(hit.secondary_moves[0])
                
                carrier_mask: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
                for empty_idx in empty_positions:
                    empty_idx_int: int = int(empty_idx)
                    if empty_idx_int != reverser:
                        carrier_mask[empty_idx_int] = True
                
                inf.add_s_reversible(p_int, carrier_mask, reverser, False)
                
                carrier_mask[p_int] = True
                for o in others:
                    o_int: int = int(o)
                    carrier_mask[o_int] = False
                    inf.add_s_reversible(o_int, carrier_mask.copy(), reverser, False)
                    carrier_mask[o_int] = True
    
    @staticmethod
    def find_t_reversible(
        pastate: PatternState,
        color: int,
        consider: npt.NDArray[np.bool_],
        inf: InferiorCells,
        patterns: IcePatternSet,
    ) -> None:
        rev_mask: npt.NDArray[np.bool_]
        hits_array: npt.NDArray[np.object_]
        rev_mask, hits_array = pastate.match_on_board(
            consider, patterns.hashed_t_reversible(color), MATCH_MODE_ALL
        )
        
        for p in np.flatnonzero(rev_mask):
            p_int: int = int(p)
            hits: npt.NDArray[np.object_] = hits_array[p_int]
            for hit_obj in hits:
                hit: PatternHit = hit_obj
                empty_positions: npt.NDArray[np.int32] = hit.empty_cells
                reverser: int = int(hit.secondary_moves[0])
                
                carrier_mask: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
                for empty_idx in empty_positions:
                    empty_idx_int: int = int(empty_idx)
                    if empty_idx_int != reverser:
                        carrier_mask[empty_idx_int] = True
                
                inf.add_s_reversible(p_int, carrier_mask, reverser, True)
    
    @staticmethod
    def find_vulnerable(
        pastate: PatternState,
        color: int,
        consider: npt.NDArray[np.bool_],
        inf: InferiorCells,
        patterns: IcePatternSet,
        find_all_killers: bool,
    ) -> None:
        matchmode: int = MATCH_MODE_ALL if find_all_killers else MATCH_MODE_FIRST
        
        vul_mask: npt.NDArray[np.bool_]
        hits_array: npt.NDArray[np.object_]
        vul_mask, hits_array = pastate.match_on_board(
            consider, patterns.hashed_vulnerable(color), matchmode
        )
        
        for p in np.flatnonzero(vul_mask):
            p_int: int = int(p)
            hits: npt.NDArray[np.object_] = hits_array[p_int]
            for hit_obj in hits:
                hit: PatternHit = hit_obj
                reverser: int = int(hit.secondary_moves[0])
                inf.add_vulnerable(p_int, {reverser})
                if not find_all_killers:
                    break
    
    @staticmethod
    def find_inferior(
        pastate: PatternState,
        color: int,
        consider: npt.NDArray[np.bool_],
        inf: InferiorCells,
        patterns: IcePatternSet,
        find_all_superiors: bool,
    ) -> None:
        matchmode: int = MATCH_MODE_ALL if find_all_superiors else MATCH_MODE_FIRST
        
        infe_mask: npt.NDArray[np.bool_]
        hits_array: npt.NDArray[np.object_]
        infe_mask, hits_array = pastate.match_on_board(
            consider, patterns.hashed_inferior(color), matchmode
        )
        
        for p in np.flatnonzero(infe_mask):
            p_int: int = int(p)
            hits: npt.NDArray[np.object_] = hits_array[p_int]
            for hit_obj in hits:
                hit: PatternHit = hit_obj
                others: npt.NDArray[np.int32] = hit.primary_moves
                superior: int = int(hit.secondary_moves[0])
                
                inf.add_inferior(p_int, {superior})
                for o in others:
                    o_int: int = int(o)
                    inf.add_inferior(o_int, {superior})
    
    @staticmethod
    def find_inferior_on_cell(
        pastate: PatternState,
        color: int,
        cell: int,
        patterns: IcePatternSet,
    ) -> npt.NDArray[np.object_]:
        hits_list: list[PatternHit] = []
        hits: npt.NDArray[np.object_] = pastate.match_on_cell(
            patterns.hashed_inferior(color), cell, MATCH_MODE_ALL
        )
        for hit_obj in hits:
            hits_list.append(hit_obj)
        
        board_state: npt.NDArray[np.int32] = pastate.board_state
        
        neighbors_raw: npt.NDArray[np.int32] = NEIGHBORS_LOOKUP[cell]
        for n_raw in neighbors_raw:
            q: int = int(n_raw)
            if q == -1:
                continue
            
            if board_state[q] != EMPTY:
                continue
            
            loc_hits: npt.NDArray[np.object_] = pastate.match_on_cell(
                patterns.hashed_inferior(color), q, MATCH_MODE_ALL
            )
            
            for hit_obj in loc_hits:
                hit: PatternHit = hit_obj
                others: npt.NDArray[np.int32] = hit.primary_moves
                if np.any(others == cell):
                    hits_list.append(hit)
        
        return np.array(hits_list, dtype=object)