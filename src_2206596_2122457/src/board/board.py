import numpy as np
import numpy.typing as npt
from typing import Optional, Dict

from src_2206596_2122457.src.ice_system.ice_engine.ice_engine import ICEngine
from src_2206596_2122457.src.group_system.groups import Groups, GroupBuilder, BLUE, RED
from src_2206596_2122457.src.ice_system.pattern_state import PatternState
from src_2206596_2122457.src.ice_system.inferior_cells import InferiorCells
from src_2206596_2122457.src.constants import PLAYER_B, PLAYER_R, PIECE_EMPTY
from src_2206596_2122457.src.vc_system.vcs.vcs import VCS
from src_2206596_2122457.src.vc_system.vcs.vc_builder_param import VCBuilderParam
from src_2206596_2122457.src.vc_system.vcs.vcs_queries import full_exists, semi_intersection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src_2206596_2122457.src.config import MCTSConfig


class Board:
    def __init__(
        self,
        board_state: npt.NDArray[np.uint8],
        ice_engine: Optional[ICEngine] = None,
        config: Optional["MCTSConfig"] = None,
    ) -> None:
        self._validate_board_state(board_state)
        self.board_state: npt.NDArray[np.uint8] = board_state.copy()
        self.n_cells: int = board_state.size
        self.ice_engine: Optional[ICEngine] = ice_engine
        self.groups: Groups = GroupBuilder.build(self.board_state)
        self.pastate: PatternState = PatternState(self.board_state)
        self.inferior_cells: InferiorCells = InferiorCells(self.n_cells)
        self._use_ice: bool = True
        self.last_compute_reverser: Optional[int] = None
        
        board_state_int: npt.NDArray[np.int_] = self.board_state.astype(np.int_)
        self._cons: Dict[int, VCS] = {
            PLAYER_B: VCS(BLUE, board_state_int),
            PLAYER_R: VCS(RED, board_state_int),
        }
        self._use_vcs: bool = True
        if config is not None:
            self._vc_builder_param = VCBuilderParam(
                and_over_edge=config.vc_and_over_edge,
                use_patterns=config.vc_use_patterns,
                use_non_edge_patterns=config.vc_use_non_edge_patterns,
                incremental_builds=config.vc_incremental_builds,
                limit_fulls=config.vc_limit_fulls,
                limit_or=config.vc_limit_or,
            )
            self._use_vcs = config.use_vcs
        else:
            self._vc_builder_param = VCBuilderParam()
    
    def _validate_board_state(self, board_state: npt.NDArray[np.uint8]) -> None:
        if board_state.ndim != 1:
            raise ValueError("board_state must be a 1-D numpy array")
        if not np.issubdtype(board_state.dtype, np.uint8):
            raise ValueError("board_state must have uint8 dtype")
        if board_state.size == 0:
            raise ValueError("board_state must not be empty")
        valid_values = {PIECE_EMPTY, PLAYER_B, PLAYER_R}
        if not np.all(np.isin(board_state, np.array(list(valid_values), dtype=np.uint8))):
            raise ValueError(f"board_state contains invalid values; must be in {valid_values}")
    
    def use_ice(self) -> bool:
        return self._use_ice
    
    def set_use_ice(self, enable: bool) -> None:
        self._use_ice = enable
    
    def use_vcs(self) -> bool:
        return self._use_vcs
    
    def set_use_vcs(self, enable: bool) -> None:
        self._use_vcs = enable
    
    def vc_builder_parameters(self) -> VCBuilderParam:
        return self._vc_builder_param
    
    def cons(self, color: int) -> VCS:
        if color not in (PLAYER_B, PLAYER_R):
            raise ValueError(f"color must be PLAYER_B ({PLAYER_B}) or PLAYER_R ({PLAYER_R}), got {color}")
        return self._cons[color]
    
    def get_inferior_cells(self) -> InferiorCells:
        return self.inferior_cells
    
    def clear_inferior_cells(self) -> None:
        self.inferior_cells.clear()
        self.inferior_cells.clear_fillin(PLAYER_B)
        self.inferior_cells.clear_fillin(PLAYER_R)
        self.last_compute_reverser = None
    
    def compute_inferior_cells(
        self,
        color_to_move: int,
        last_move: Optional[int] = None,
        only_around_last_move: bool = False,
    ) -> Optional[int]:
        if not self._use_ice or self.ice_engine is None:
            self.last_compute_reverser = None
            return None
        
        self._validate_color(color_to_move)
        self._validate_last_move(last_move)
        
        last_move_arg: int = last_move if last_move is not None else -1
        
        reverser_index: int = self.ice_engine.compute_inferior_cells(
            color_to_move,
            self.groups,
            self.pastate,
            self.inferior_cells,
            last_move=last_move_arg,
            only_around_last_move=only_around_last_move,
        )
        
        reverser_index_sanitized: Optional[int] = self._sanitize_reverser(reverser_index)
        self.last_compute_reverser = reverser_index_sanitized
        
        return reverser_index_sanitized
    
    def compute_all(
        self,
        color_to_move: int,
        last_move: Optional[int] = None,
        add_fillin: bool = False,
        only_around_last_move: bool = False,
    ) -> Optional[int]:
        self.pastate.update_all_cells()
                
        self.groups = GroupBuilder.build(self.board_state)
        self.inferior_cells.clear()
        self.inferior_cells.clear_fillin(PLAYER_B)
        self.inferior_cells.clear_fillin(PLAYER_R)
        
        reverser: Optional[int] = self.compute_inferior_cells(
            color_to_move, last_move, only_around_last_move
        )
        
        if self._use_vcs:
            self._build_vcs()
        
        if add_fillin:
            fillin_b: npt.NDArray[np.bool_] = self.inferior_cells.fillin(PLAYER_B)
            fillin_r: npt.NDArray[np.bool_] = self.inferior_cells.fillin(PLAYER_R)
            
            self.board_state[fillin_b] = PLAYER_B
            self.board_state[fillin_r] = PLAYER_R
            
            if self._use_vcs:
                self._build_vcs()
            
            changed_mask: npt.NDArray[np.bool_] = fillin_b | fillin_r
            changed_indices: npt.NDArray[np.int_] = np.flatnonzero(changed_mask)
            if len(changed_indices) > 0:
                self.pastate.update_changed_cells(changed_indices)
        
        if reverser is not None:
            fillin_b_check: npt.NDArray[np.bool_] = self.inferior_cells.fillin(PLAYER_B)
            fillin_r_check: npt.NDArray[np.bool_] = self.inferior_cells.fillin(PLAYER_R)
            if fillin_b_check[reverser] or fillin_r_check[reverser]:
                reverser = None
        
        return reverser
    
    
    def _validate_color(self, color_to_move: int) -> None:
        if color_to_move not in (PLAYER_B, PLAYER_R):
            raise ValueError(f"color_to_move must be PLAYER_B ({PLAYER_B}) or PLAYER_R ({PLAYER_R}), got {color_to_move}")
    
    def _validate_last_move(self, last_move: Optional[int]) -> None:
        if last_move is not None and last_move != -1:
            if last_move < 0 or last_move >= self.n_cells:
                raise IndexError(f"last_move must be in range [0, {self.n_cells - 1}], -1, or None, got {last_move}")
    
    def _sanitize_reverser(self, reverser_index: int) -> Optional[int]:
        if reverser_index == -1:
            return None
        
        if reverser_index < 0 or reverser_index >= self.n_cells:
            return None
        
        return reverser_index
    
    def _build_vcs(self) -> None:
        if not self._use_vcs:
            return
        
        board_state_int: npt.NDArray[np.int_] = self.board_state.astype(np.int_)
        for color in (PLAYER_B, PLAYER_R):
            vcs = self._cons[color]
            vcs.set_table(board_state_int)
            vcs.build(self._vc_builder_param, self.groups)
    
    def _revert_vcs(self) -> None:
        if not self._use_vcs:
            return
        
        for color in (PLAYER_B, PLAYER_R):
            self._cons[color].reset()
    
    def get_mustplay(self, color: int) -> npt.NDArray[np.bool_]:
        other_color: int = PLAYER_R if color == PLAYER_B else PLAYER_B
        other_vcs: VCS = self.cons(other_color)
        
        if full_exists(other_vcs):
            return np.zeros(self.n_cells, dtype=bool)
        
        empty_mask: npt.NDArray[np.bool_] = self.board_state == PIECE_EMPTY
        semi_int: npt.NDArray[np.bool_] = semi_intersection(other_vcs)
        
        return empty_mask & semi_int