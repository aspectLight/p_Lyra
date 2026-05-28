import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

from src_2206596_2122457.src.constants import BOARD_SIZE
from src_2206596_2122457.src.group_system.groups import Groups, BLUE, RED, EMPTY
from src_2206596_2122457.src.vc_system.vc_pattern import VcPatternManager
from src_2206596_2122457.src.vc_system.vcs.vc_builder_param import VCBuilderParam
from src_2206596_2122457.src.vc_system.vcs.and_list import AndList
from src_2206596_2122457.src.vc_system.vcs.value_objects import Full

if TYPE_CHECKING:
    from src_2206596_2122457.src.vc_system.vcs.vcs import VCS


def compute_captured_sets(
    vcs: "VCS",
    board_state: NDArray[np.int_],
) -> None:
    empty_indices: NDArray[np.int_] = np.flatnonzero(board_state == EMPTY)
    
    pattern_manager: VcPatternManager = VcPatternManager(BOARD_SIZE, BOARD_SIZE)
    color: int = vcs.get_color()
    vc_patterns = pattern_manager.get_patterns_for_color(color)
    
    m_captured_set = vcs.get_m_captured_set()
    
    for cell_idx in empty_indices:
        captured: NDArray[np.bool_] = np.zeros(len(board_state), dtype=bool)
        
        for pattern in vc_patterns:
            if pattern.matches(color, board_state):
                ep0, ep1 = pattern.endpoints
                not_opponent: NDArray[np.bool_] = pattern.not_opponent
                
                if ep0 == cell_idx or ep1 == cell_idx:
                    for i in np.flatnonzero(not_opponent):
                        if i < len(board_state):
                            captured[i] = True
        
        m_captured_set[cell_idx] = captured


def add_base_vcs(
    vcs: "VCS",
    groups: Groups,
    board_state: NDArray[np.int_],
) -> None:
    empty_mask: NDArray[np.bool_] = board_state == EMPTY
    
    color: int = vcs.get_color()
    for group in groups.groups:
        if group.color != color and group.color != EMPTY:
            continue
        
        captain: int = group.captain
        neighbors: NDArray[np.int_] = group.neighbors
        
        for neighbor_idx in np.where(neighbors)[0]:
            if neighbor_idx >= len(board_state):
                continue
            
            if empty_mask[neighbor_idx]:
                    if groups.captain_of(neighbor_idx) == neighbor_idx:
                        if captain != neighbor_idx:
                            empty_carrier: NDArray[np.bool_] = np.zeros(
                                len(board_state), dtype=bool
                            )
                            
                            try_add_full(vcs, captain, neighbor_idx, empty_carrier)


def add_pattern_vcs(
    vcs: "VCS",
    groups: Groups,
    board_state: NDArray[np.int_],
) -> None:
    m_param = vcs.get_m_param()
    if not m_param or not m_param.use_patterns:
        return
    
    pattern_manager: VcPatternManager = VcPatternManager(BOARD_SIZE, BOARD_SIZE)
    color: int = vcs.get_color()
    opponent_color: int = vcs.get_opponent_color()
    vc_patterns = pattern_manager.get_patterns_for_color(color)
    
    for pattern in vc_patterns:
        if not m_param.use_non_edge_patterns:
            ep0, ep1 = pattern.endpoints
            row0: int = ep0 // BOARD_SIZE
            col0: int = ep0 % BOARD_SIZE
            row1: int = ep1 // BOARD_SIZE
            col1: int = ep1 % BOARD_SIZE
            
            is_ep0_edge: bool = (
                (color == BLUE and (row0 == 0 or row0 == BOARD_SIZE - 1))
                or (color == RED and (col0 == 0 or col0 == BOARD_SIZE - 1))
            )
            is_ep1_edge: bool = (
                (color == BLUE and (row1 == 0 or row1 == BOARD_SIZE - 1))
                or (color == RED and (col1 == 0 or col1 == BOARD_SIZE - 1))
            )
            
            if not is_ep0_edge and not is_ep1_edge:
                continue
        
        if pattern.matches(color, board_state):
            ep0, ep1 = pattern.endpoints
            not_opponent: NDArray[np.bool_] = pattern.not_opponent
            
            carrier: NDArray[np.bool_] = not_opponent.copy()
            
            opponent_mask: NDArray[np.bool_] = board_state == opponent_color
            carrier = carrier & ~opponent_mask
            
            if ep0 < len(carrier):
                carrier[ep0] = False
            if ep1 < len(carrier):
                carrier[ep1] = False
            
            x: int = groups.captain_of(ep0)
            y: int = groups.captain_of(ep1)
            
            if x == y:
                continue
            
            try_add_full(vcs, x, y, carrier)


def try_add_full(
    vcs: "VCS",
    x: int,
    y: int,
    carrier: NDArray[np.bool_],
) -> bool:
    if x >= len(carrier) or y >= len(carrier):
        return False
    
    table: NDArray[np.int_] = vcs.get_table()
    opponent_color: int = vcs.get_opponent_color()
    opponent_mask: NDArray[np.bool_] = table == opponent_color
    if np.any(carrier & opponent_mask):
        return False
    
    fulls: AndList = vcs.get_full_carriers_at(x, y)
    m_param = vcs.get_m_param()
    limit: bool = m_param.limit_fulls if m_param else False
    
    if fulls.try_add(carrier, limit):
        m_fulls_and_queue = vcs.get_m_fulls_and_queue()
        m_fulls_and_queue.append(Full(x=x, y=y, carrier=carrier))
        return True
    
    return False


def build(
    vcs: "VCS",
    param: VCBuilderParam,
    groups: Groups,
    board_state: NDArray[np.int_],
) -> None:
    vcs.set_m_param(param)
    vcs.set_m_groups(groups)
    vcs.set_table(board_state)
    
    vcs.reset()
    
    compute_captured_sets(vcs, board_state)
    add_base_vcs(vcs, groups, board_state)
    
    if param.use_patterns:
        add_pattern_vcs(vcs, groups, board_state)
    
    from src_2206596_2122457.src.vc_system.vcs.vcs_search import do_search
    do_search(vcs)

