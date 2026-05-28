import numpy as np
import numpy.typing as npt
from typing import Set

from src_2206596_2122457.src.ice_system.pattern_state import PatternState
from src_2206596_2122457.src.ice_system.ice_pattern_set import IcePatternSet
from src_2206596_2122457.src.ice_system.inferior_cells import InferiorCells
from src_2206596_2122457.src.ice_system.ice_engine.ice_engine_helpers import IceEngineHelpers
from src_2206596_2122457.src.ice_system.ice_engine.ice_engine_pattern_finders import IceEnginePatternFinders
from src_2206596_2122457.src.group_system.dead_region import DeadRegion
from src_2206596_2122457.src.group_system.groups import Groups, EMPTY, BLUE, RED, TOTAL_CELLS
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP


class IceEngineDeadVulnerable:
    
    @staticmethod
    def clique_cutset_dead(
        color: int,
        groups: Groups,
        pastate: PatternState,
        inf: InferiorCells,
        find_three_sided: bool,
    ) -> int:
        not_reachable: npt.NDArray[np.bool_] = DeadRegion.compute_group_dead_regions(groups)
        
        if find_three_sided:
            not_reachable |= DeadRegion.compute_dead_regions_multi_group_interaction(groups)
        
        if not_reachable.any():
            inf.add_fillin(color, not_reachable)
            pastate.board_state[not_reachable] = color
            pastate.update_changed_cells(np.flatnonzero(not_reachable))
        
        return int(not_reachable.sum())
    
    @staticmethod
    def fill_in_vulnerable(
        color: int,
        groups: Groups,
        pastate: PatternState,
        inf: InferiorCells,
        patterns: IcePatternSet,
        find_all_killers: bool,
    ) -> int:
        inf.clear_vulnerable()
        inf.clear_s_reversible()
        
        opponent_color: int = RED if color == BLUE else BLUE
        IceEngineDeadVulnerable.use_graph_theory_to_find_dead_vulnerable(
            opponent_color, groups, pastate, inf
        )
        
        consider_mask: npt.NDArray[np.bool_] = pastate.board_state == EMPTY
        IceEnginePatternFinders.find_vulnerable(
            pastate, opponent_color, consider_mask, inf, patterns, find_all_killers
        )
        
        fillin_mask: npt.NDArray[np.bool_] = inf.find_presimplicial_pairs()
        
        if fillin_mask.any():
            inf.add_fillin(color, fillin_mask)
            pastate.board_state[fillin_mask] = color
            pastate.update_changed_cells(np.flatnonzero(fillin_mask))
        
        return int(fillin_mask.sum())
    
    @staticmethod
    def use_graph_theory_to_find_dead_vulnerable(
        color: int,
        groups: Groups,
        pastate: PatternState,
        inf: InferiorCells,
    ) -> None:
        board_state: npt.NDArray[np.int32] = pastate.board_state
        consider: npt.NDArray[np.bool_] = board_state == EMPTY
        
        for p in np.flatnonzero(consider):
            p_int: int = int(p)
            
            enbs: Set[int] = set()
            cnbs: Set[int] = set()
            empty_adj_to_group: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
            adj_to_edge: bool = False
            edge_nbr: int = -1
            
            neighbors_raw: npt.NDArray[np.int32] = NEIGHBORS_LOOKUP[p_int]
            for n_raw in neighbors_raw:
                n: int = int(n_raw)
                if n == -1:
                    continue
                
                ncolor: int = int(board_state[n])
                if ncolor == EMPTY:
                    enbs.add(n)
                elif ncolor == color:
                    cap: int = groups.captain_of(n)
                    group_obj = groups.get_group(n)
                    adj: npt.NDArray[np.bool_] = group_obj.neighbors.copy()
                    adj[p_int] = False
                    
                    if IceEngineHelpers.is_color_edge(cap, color):
                        adj_to_edge = True
                        edge_nbr = cap
                        cnbs.add(cap)
                        empty_adj_to_group |= adj
                    elif int(adj.sum()) == 1:
                        empty_single: int = int(np.flatnonzero(adj)[0])
                        enbs.add(empty_single)
                    elif int(adj.sum()) >= 2:
                        cnbs.add(cap)
                        empty_adj_to_group |= adj
            
            to_remove: Set[int] = set()
            for en in enbs:
                if empty_adj_to_group[en]:
                    to_remove.add(en)
            enbs -= to_remove
            
            if len(enbs) + len(cnbs) <= 1:
                inf.add_vulnerable(p_int, enbs if enbs else cnbs)
            elif adj_to_edge or len(cnbs) >= 2:
                if len(enbs) >= 2:
                    continue
                
                if len(cnbs) == 1:
                    if len(enbs) == 1:
                        inf.add_vulnerable(p_int, enbs)
                else:
                    killers_bs: Set[int] = set()
                    
                    for i_cn in cnbs:
                        if adj_to_edge and i_cn != edge_nbr:
                            continue
                        
                        i_group = groups.get_group(i_cn)
                        group_neighbors: npt.NDArray[np.bool_] = i_group.neighbors.copy()
                        remaining_nbs: npt.NDArray[np.bool_] = (
                            empty_adj_to_group & ~group_neighbors
                        )
                        
                        if int(remaining_nbs.sum()) == 0:
                            if len(enbs) > 0:
                                killers_bs |= enbs
                        elif int(remaining_nbs.sum()) == 1 and len(enbs) == 0:
                            killers_bs.add(int(np.flatnonzero(remaining_nbs)[0]))
                    
                    if killers_bs:
                        inf.add_vulnerable(p_int, killers_bs)
            elif len(enbs) + len(cnbs) >= 4:
                pass
            elif len(cnbs) == 1:
                if len(enbs) > 1:
                    continue
                
                inf.add_vulnerable(p_int, enbs)
                
                if int(empty_adj_to_group.sum()) == 2:
                    omit: int = list(enbs)[0] if enbs else -1
                    for i_empty in np.flatnonzero(empty_adj_to_group):
                        i_int: int = int(i_empty)
                        enbs.add(i_int)
                    
                    vn: list[int] = sorted(list(enbs))
                    for _, ex in enumerate(vn):
                        if ex == omit:
                            continue
                        vn_copy = vn.copy()
                        if IceEngineHelpers.verify_complete_adjacency(
                            np.array(vn_copy, dtype=np.int32), ex
                        ):
                            inf.add_vulnerable(p_int, {ex})
            else:
                vn = sorted(list(enbs))
                if IceEngineHelpers.verify_complete_adjacency(np.array(vn, dtype=np.int32)):
                    inf.add_vulnerable(p_int, set())
                else:
                    for _, ex in enumerate(vn):
                        vn_copy = vn.copy()
                        if IceEngineHelpers.verify_complete_adjacency(
                            np.array(vn_copy, dtype=np.int32), ex
                        ):
                            inf.add_vulnerable(p_int, {ex})