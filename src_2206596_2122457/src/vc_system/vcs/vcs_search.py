import numpy as np
from typing import TYPE_CHECKING
from numpy.typing import NDArray

from src_2206596_2122457.src.group_system.groups import EMPTY
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP
from src_2206596_2122457.src.vc_system.vcs.vcs_build import try_add_full
from src_2206596_2122457.src.vc_system.vc_or import vc_or
from src_2206596_2122457.src.vc_system.carrier_list import CarrierList

if TYPE_CHECKING:
    from src_2206596_2122457.src.vc_system.vcs.vcs import VCS


def get_neighbors(point: int) -> NDArray[np.int32]:
    neighbor_data: NDArray[np.int32] = NEIGHBORS_LOOKUP[point]
    mask: NDArray[np.bool_] = neighbor_data != -1
    return neighbor_data[mask]


def do_search(vcs: "VCS") -> None:
    m_fulls_and_queue = vcs.get_m_fulls_and_queue()
    m_semis_or_queue = vcs.get_m_semis_or_queue()
    while m_fulls_and_queue or m_semis_or_queue:
        if m_fulls_and_queue:
            full = m_fulls_and_queue.popleft()
            and_full(vcs, full.x, full.y, full.carrier)
        elif m_semis_or_queue:
            ends = m_semis_or_queue.popleft()
            or_semis(vcs, ends.x, ends.y)


def and_full(
    vcs: "VCS",
    x: int,
    y: int,
    carrier: NDArray[np.bool_],
) -> None:
    m_groups = vcs.get_m_groups()
    if not m_groups:
        return
    
    fulls = vcs.get_full_carriers_at(x, y)
    fulls.try_set_processed(carrier)
    
    m_captured_set = vcs.get_m_captured_set()
    n_positions = vcs.get_n_positions()
    x_captured: NDArray[np.bool_] = (
        m_captured_set.get(x, np.zeros(n_positions, dtype=bool))
    )
    y_captured: NDArray[np.bool_] = (
        m_captured_set.get(y, np.zeros(n_positions, dtype=bool))
    )
    
    xy_captured: NDArray[np.bool_] = x_captured | y_captured
    
    z_candidates: NDArray[np.int32] = get_neighbors(y)
    
    table: NDArray[np.int_] = vcs.get_table()
    opponent_color: int = vcs.get_opponent_color()
    for z in z_candidates:
        if z < 0 or z >= len(table):
            continue
        
        z_color: int = int(table[z])
        
        if z_color == opponent_color:
            continue
        
        z_captured: NDArray[np.bool_] = (
            m_captured_set.get(z, np.zeros(n_positions, dtype=bool))
        )
        
        if m_groups:
            z_captain: int = m_groups.captain_of(z)
        else:
            z_captain = z
        
        zy_full_list = vcs.get_full_carriers_at(z_captain, y)
        
        if zy_full_list.is_empty():
            continue
        
        zy_iter = zy_full_list.iterator()
        while zy_iter:
            zy_carrier: NDArray[np.bool_] = zy_iter.carrier()
            
            new_carrier: NDArray[np.bool_] = carrier | zy_carrier
            
            opponent_mask: NDArray[np.bool_] = table == opponent_color
            if np.any(new_carrier & opponent_mask):
                zy_iter.__next__()
                continue
            
            if z_color == EMPTY:
                and_full_empty_full(
                    vcs, x, z_captain, new_carrier, xy_captured | z_captured
                )
            else:
                and_full_stone_full(
                    vcs, x, z_captain, new_carrier
                )
            
            zy_iter.__next__()


def and_full_empty_full(
    vcs: "VCS",
    x: int,
    z: int,
    carrier: NDArray[np.bool_],
    xz_captured_set: NDArray[np.bool_],
) -> bool:
    if x == z:
        return False
    
    if try_add_full(vcs, x, z, carrier):
        return True
    
    m_param = vcs.get_m_param()
    if m_param and m_param.and_over_edge:
        intersection: NDArray[np.bool_] = carrier & xz_captured_set
        if np.any(intersection):
            return try_add_full(vcs, x, z, carrier)
    
    return False


def and_full_stone_full(
    vcs: "VCS",
    x: int,
    z: int,
    carrier: NDArray[np.bool_],
) -> bool:
    if x == z:
        return False
    
    return try_add_full(vcs, x, z, carrier)


def or_semis(vcs: "VCS", x: int, y: int) -> None:
    m_groups = vcs.get_m_groups()
    if not m_groups:
        return
    
    semis = vcs.get_semi_carriers_at(x, y)
    
    if semis.is_empty():
        return
    
    m_captured_set = vcs.get_m_captured_set()
    n_positions = vcs.get_n_positions()
    x_captured: NDArray[np.bool_] = (
        m_captured_set.get(x, np.zeros(n_positions, dtype=bool))
    )
    y_captured: NDArray[np.bool_] = (
        m_captured_set.get(y, np.zeros(n_positions, dtype=bool))
    )
    
    semis_list = CarrierList()
    semis_iter = semis.iterator()
    while semis_iter:
        semis_list.add_new(semis_iter.carrier().copy())
        semis_iter.__next__()
    
    fulls_list = vcs.get_full_carriers_at(x, y)
    fulls_carrier_list = CarrierList()
    fulls_iter = fulls_list.iterator()
    while fulls_iter:
        fulls_carrier_list.add_new(fulls_iter.carrier().copy())
        fulls_iter.__next__()
    
    result_carriers = vc_or(
        semis_list,
        fulls_carrier_list,
        x_captured,
        y_captured,
    )
    
    for result_carrier in result_carriers:
        try_add_full(vcs, x, y, result_carrier)
    
    semis.mark_all_processed()

