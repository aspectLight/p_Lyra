import numpy as np
import numpy.typing as npt
from typing import Optional

from src_2206596_2122457.src.ice_system.ice_pattern_set import IcePatternSet
from src_2206596_2122457.src.ice_system.pattern_state import PatternState, PatternHit
from src_2206596_2122457.src.ice_system.ice_engine.ice_fillin_modes import FillinMode
from src_2206596_2122457.src.ice_system.ice_engine.ice_engine_helpers import IceEngineHelpers
from src_2206596_2122457.src.ice_system.ice_engine.ice_engine_pattern_finders import IceEnginePatternFinders
from src_2206596_2122457.src.ice_system.ice_engine.ice_engine_reversible import IceEngineReversible
from src_2206596_2122457.src.ice_system.ice_engine.ice_engine_dead_vulnerable import IceEngineDeadVulnerable
from src_2206596_2122457.src.group_system.groups import Groups, GroupBuilder, EMPTY, BLUE, RED, TOTAL_CELLS
from src_2206596_2122457.src.ice_system.inferior_cells import InferiorCells
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP
from src_2206596_2122457.src.ice_system.ice_constants import MATCH_MODE_FIRST

class ICEngine:
    
    def __init__(self) -> None:
        self.m_find_presimplicial_pairs: bool = False
        self.m_find_all_pattern_killers: bool = False
        self.m_find_all_pattern_superiors: bool = True
        self.m_find_three_sided_dead_regions: bool = False
        self.m_iterative_dead_regions: bool = False
        self.m_use_capture: bool = True
        self.m_find_reversible: bool = True
        self.m_use_s_reversible_as_reversible: bool = False
        
        self.m_patterns: IcePatternSet = IcePatternSet()
    
    def load_patterns(self) -> None:
        self.m_patterns = IcePatternSet()
    
    def find_presimplicial_pairs(self) -> bool:
        return self.m_find_presimplicial_pairs
    
    def set_find_presimplicial_pairs(self, enable: bool) -> None:
        self.m_find_presimplicial_pairs = enable
    
    def find_all_pattern_killers(self) -> bool:
        return self.m_find_all_pattern_killers
    
    def set_find_all_pattern_killers(self, enable: bool) -> None:
        self.m_find_all_pattern_killers = enable
    
    def find_all_pattern_superiors(self) -> bool:
        return self.m_find_all_pattern_superiors
    
    def set_find_all_pattern_superiors(self, enable: bool) -> None:
        self.m_find_all_pattern_superiors = enable
    
    def find_three_sided_dead_regions(self) -> bool:
        return self.m_find_three_sided_dead_regions
    
    def set_find_three_sided_dead_regions(self, enable: bool) -> None:
        self.m_find_three_sided_dead_regions = enable
    
    def iterative_dead_regions(self) -> bool:
        return self.m_iterative_dead_regions
    
    def set_iterative_dead_regions(self, enable: bool) -> None:
        self.m_iterative_dead_regions = enable
    
    def use_capture(self) -> bool:
        return self.m_use_capture
    
    def set_use_capture(self, enable: bool) -> None:
        self.m_use_capture = enable
    
    def find_reversible(self) -> bool:
        return self.m_find_reversible
    
    def set_find_reversible(self, enable: bool) -> None:
        self.m_find_reversible = enable
    
    def use_s_reversible_as_reversible(self) -> bool:
        return self.m_use_s_reversible_as_reversible
    
    def set_use_s_reversible_as_reversible(self, enable: bool) -> None:
        self.m_use_s_reversible_as_reversible = enable
    
    def compute_inferior_cells(
        self,
        color: int,
        groups: Groups,
        pastate: PatternState,
        inf: InferiorCells,
        last_move: int = -1,
        only_around_last_move: bool = False,
    ) -> int:
        opponent_color: int = RED if color == BLUE else BLUE
        reverser: int = -1
        
        if self.m_find_reversible and last_move != -1:
            reverser = IceEngineReversible.is_reversible(
                pastate, opponent_color, last_move, self.m_patterns,
                self.m_use_s_reversible_as_reversible
            )
        
        if only_around_last_move and last_move != -1:
            self.compute_fillin(
                groups, pastate, inf, color, FillinMode.BICOLOR,
                consider=None, last_move=last_move, clear_inf=True
            )
        else:
            self.compute_fillin(
                groups, pastate, inf, color, FillinMode.BICOLOR,
                consider=None, last_move=-1, clear_inf=True
            )
        
        consider_mask: npt.NDArray[np.bool_] = pastate.board_state == EMPTY
        
        IceEnginePatternFinders.find_s_reversible(
            pastate, color, consider_mask, inf, self.m_patterns,
            self.m_find_all_pattern_killers
        )
        IceEnginePatternFinders.find_t_reversible(
            pastate, color, consider_mask, inf, self.m_patterns
        )
        IceEnginePatternFinders.find_inferior(
            pastate, color, consider_mask, inf, self.m_patterns,
            self.m_find_all_pattern_superiors
        )
        
        return reverser
    
    def compute_fillin(
        self,
        groups: Groups,
        pastate: PatternState,
        inf: InferiorCells,
        color: int,
        mode: FillinMode,
        consider: Optional[npt.NDArray[np.bool_]] = None,
        last_move: int = -1,
        clear_inf: bool = True,
    ) -> int:
        mode = IceEngineHelpers.turn_off_capture(mode) if not self.m_use_capture else mode
        
        if consider is None:
            if last_move != -1:
                consider_mask: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
                neighbors_raw: npt.NDArray[np.int32] = NEIGHBORS_LOOKUP[last_move]
                for n in neighbors_raw:
                    if n != -1:
                        consider_mask[int(n)] = True
                consider_mask &= (pastate.board_state == EMPTY)
            else:
                consider_mask = pastate.board_state == EMPTY
        else:
            consider_mask = consider.copy()
        
        if clear_inf:
            inf.clear()
        
        count: int = 0
        opponent_color: int = RED if color == BLUE else BLUE
        
        while True:
            count += self._compute_pattern_fillin(pastate, inf, color, mode, consider_mask)
            
            if self.m_find_presimplicial_pairs:
                loc_count: int = IceEngineDeadVulnerable.fill_in_vulnerable(
                    color, groups, pastate, inf, self.m_patterns,
                    self.m_find_all_pattern_killers
                )
                if IceEngineHelpers.uses_capture(mode):
                    loc_count += IceEngineDeadVulnerable.fill_in_vulnerable(
                        opponent_color, groups, pastate, inf, self.m_patterns,
                        self.m_find_all_pattern_killers
                    )
                if loc_count != 0:
                    count += loc_count
                    consider_mask = pastate.board_state == EMPTY
                    continue
            
            if self.m_iterative_dead_regions:
                loc_count = IceEngineDeadVulnerable.clique_cutset_dead(
                    color, groups, pastate, inf, self.m_find_three_sided_dead_regions
                )
                if loc_count != 0:
                    count += loc_count
                    consider_mask = pastate.board_state == EMPTY
                    continue
            
            break
        
        if not self.m_iterative_dead_regions:
            count += IceEngineDeadVulnerable.clique_cutset_dead(
                color, groups, pastate, inf, self.m_find_three_sided_dead_regions
            )
        
        captured: npt.NDArray[np.bool_] = inf.fillin(opponent_color)
        if IceEngineHelpers.is_monocolor_using_capture(mode) and captured.any():
            inf.clear_fillin(opponent_color)
            pastate.board_state[captured] = EMPTY
            pastate.update_changed_cells(np.flatnonzero(captured))
            count -= int(captured.sum())
            
            consider_neighbors: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
            for cap_idx in np.flatnonzero(captured):
                neighbors_raw = NEIGHBORS_LOOKUP[int(cap_idx)]
                for n in neighbors_raw:
                    if n != -1 and pastate.board_state[int(n)] == EMPTY:
                        consider_neighbors[int(n)] = True
            
            groups_new: Groups = GroupBuilder.build(pastate.board_state)
            groups.groups = groups_new.groups
            groups.group_index = groups_new.group_index
            
            loc_count = self.compute_fillin(
                groups, pastate, inf, color, FillinMode.MONOCOLOR,
                consider=consider_neighbors, last_move=-1, clear_inf=False
            )
            count += loc_count
        elif count:
            groups_new: Groups = GroupBuilder.build(pastate.board_state)
            groups.groups = groups_new.groups
            groups.group_index = groups_new.group_index
        
        return count
    
    def _compute_pattern_fillin(
        self,
        pastate: PatternState,
        inf: InferiorCells,
        color: int,
        mode: FillinMode,
        consider: npt.NDArray[np.bool_],
    ) -> int:
        board_state: npt.NDArray[np.int32] = pastate.board_state
        consider_mask: npt.NDArray[np.bool_] = consider.copy()
        count: int = 0
        opponent_color: int = RED if color == BLUE else BLUE
        
        while consider_mask.any():
            indices: npt.NDArray[np.intp] = np.flatnonzero(consider_mask)
            for p_raw in indices:
                p: int = int(p_raw)
                
                if not consider_mask[p]:
                    continue
                
                consider_mask[p] = False
                
                fillin_blue: npt.NDArray[np.bool_] = inf.fillin(BLUE)
                fillin_red: npt.NDArray[np.bool_] = inf.fillin(RED)
                if fillin_blue[p] or fillin_red[p]:
                    continue
                
                hits: npt.NDArray[np.object_] = pastate.match_on_cell(
                    self.m_patterns.hashed_e_fillin, p, MATCH_MODE_FIRST
                )
                
                if len(hits) > 0:
                    c: int = IceEngineHelpers.pick_color(board_state, color, p, mode)
                    fillin_mask: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
                    fillin_mask[p] = True
                    inf.add_fillin(c, fillin_mask)
                    board_state[p] = c
                    pastate.update_cell(p)
                    IceEngineHelpers.add_neighbors_to_consider(p, board_state, consider_mask)
                    count += 1
                    continue
                
                to_fill_colors: list[int] = [color]
                if IceEngineHelpers.uses_capture(mode):
                    to_fill_colors.append(opponent_color)
                
                for c in to_fill_colors:
                    if c == color or IceEngineHelpers.bicolor(mode):
                        hits = pastate.match_on_cell(
                            self.m_patterns.hashed_fillin(c), p, MATCH_MODE_FIRST
                        )
                    else:
                        hits = pastate.match_on_cell(
                            self.m_patterns.hashed_captured(c), p, MATCH_MODE_FIRST
                        )
                    
                    if len(hits) > 0:
                        hit: PatternHit = hits[0]
                        
                        fillin_mask_p: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
                        fillin_mask_p[p] = True
                        inf.add_fillin(c, fillin_mask_p)
                        board_state[p] = c
                        pastate.update_cell(p)
                        IceEngineHelpers.add_neighbors_to_consider(p, board_state, consider_mask)
                        count += 1
                        
                        others: npt.NDArray[np.int32] = hit.primary_moves
                        for it in others:
                            it_int: int = int(it)
                            consider_mask[it_int] = False
                            fillin_mask_it: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
                            fillin_mask_it[it_int] = True
                            inf.add_fillin(c, fillin_mask_it)
                            board_state[it_int] = c
                            pastate.update_cell(it_int)
                            IceEngineHelpers.add_neighbors_to_consider(it_int, board_state, consider_mask)
                            count += 1
                        
                        if IceEngineHelpers.uses_capture(mode):
                            opps: npt.NDArray[np.int32] = hit.secondary_moves
                            for it in opps:
                                it_int: int = int(it)
                                consider_mask[it_int] = False
                                fillin_mask_opp: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
                                fillin_mask_opp[it_int] = True
                                inf.add_fillin(opponent_color, fillin_mask_opp)
                                board_state[it_int] = opponent_color
                                pastate.update_cell(it_int)
                                IceEngineHelpers.add_neighbors_to_consider(it_int, board_state, consider_mask)
                                count += 1
                        
                        break
        
        return count
    
    def is_reversible(
        self,
        pastate: PatternState,
        color: int,
        p: int,
    ) -> int:
        return IceEngineReversible.is_reversible(
            pastate, color, p, self.m_patterns,
            self.m_use_s_reversible_as_reversible
        )
    
    def find_inferior_on_cell(
        self,
        pastate: PatternState,
        color: int,
        cell: int,
    ) -> npt.NDArray[np.object_]:
        return IceEnginePatternFinders.find_inferior_on_cell(
            pastate, color, cell, self.m_patterns
        )


class IceUtil:
    
    @staticmethod
    def update(out_inf: InferiorCells, in_inf: InferiorCells) -> None:
        out_inf.clear_vulnerable()
        out_inf.clear_s_reversible()
        out_inf.clear_inferior()
        
        out_inf.add_vulnerable_from(in_inf)
        out_inf.add_s_reversible_from(in_inf)
        out_inf.add_inferior_from(in_inf)
        
        for color in (BLUE, RED):
            fillin_in: npt.NDArray[np.bool_] = in_inf.fillin(color)
            fillin_mask: npt.NDArray[np.bool_] = np.zeros(TOTAL_CELLS, dtype=bool)
            fillin_mask |= fillin_in
            out_inf.add_fillin(color, fillin_mask)