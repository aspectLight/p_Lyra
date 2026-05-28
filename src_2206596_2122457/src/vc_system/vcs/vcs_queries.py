import numpy as np
from typing import Optional, TYPE_CHECKING
from numpy.typing import NDArray

from src_2206596_2122457.src.vc_system.carrier_list import CarrierList

if TYPE_CHECKING:
    from src_2206596_2122457.src.vc_system.vcs.vcs import VCS


def get_smallest_semis_union(vcs: "VCS") -> NDArray[np.bool_]:
    n_positions: int = vcs.get_n_positions()
    result: NDArray[np.bool_] = np.zeros(n_positions, dtype=bool)
    
    m_semis = vcs.get_m_semis()
    for semis in m_semis.values():
        if semis.is_empty():
            continue
        
        semis_union = semis.get_greedy_union()
        result |= semis_union
    
    return result


def smallest_full_carrier(vcs: "VCS") -> Optional[NDArray[np.bool_]]:
    n_positions: int = vcs.get_n_positions()
    min_size: int = n_positions + 1
    result: Optional[NDArray[np.bool_]] = None
    
    m_fulls = vcs.get_m_fulls()
    for fulls in m_fulls.values():
        if fulls.is_empty():
            continue
        
        iter_obj = fulls.iterator()
        while iter_obj:
            carrier = iter_obj.carrier()
            size = int(carrier.sum())
            
            if size < min_size:
                min_size = size
                result = carrier.copy()
            
            iter_obj.__next__()
    
    return result


def full_adjacent(vcs: "VCS", x: int, y: int) -> int:
    m_fulls = vcs.get_m_fulls()
    fulls = m_fulls.get((x, y))
    
    if not fulls or fulls.is_empty():
        return -1
    
    iter_obj = fulls.iterator()
    if not iter_obj:
        return -1
    
    carrier = iter_obj.carrier()
    if carrier.sum() == 0:
        return 0
    
    return 1


def smallest_semi_carrier(vcs: "VCS") -> Optional[NDArray[np.bool_]]:
    n_positions: int = vcs.get_n_positions()
    min_size: int = n_positions + 1
    result: Optional[NDArray[np.bool_]] = None
    
    m_semis = vcs.get_m_semis()
    for semis in m_semis.values():
        if semis.is_empty():
            continue
        
        iter_obj = semis.iterator()
        while iter_obj:
            carrier = iter_obj.carrier()
            size = int(carrier.sum())
            
            if size < min_size:
                min_size = size
                result = carrier.copy()
            
            iter_obj.__next__()
    
    return result


def semi_key(
    vcs: "VCS",
    carrier: NDArray[np.bool_],
) -> Optional[int]:
    table: NDArray[np.int_] = vcs.get_table()
    color: int = vcs.get_color()
    empty_mask: NDArray[np.bool_] = table == 0
    carrier_empty: NDArray[np.bool_] = carrier & empty_mask
    
    empty_indices: NDArray[np.int_] = np.flatnonzero(carrier_empty)
    
    if len(empty_indices) > 0:
        return int(empty_indices[0])
    
    player_mask: NDArray[np.bool_] = table == color
    carrier_player: NDArray[np.bool_] = carrier & player_mask
    
    player_indices: NDArray[np.int_] = np.flatnonzero(carrier_player)
    
    if len(player_indices) > 0:
        return int(player_indices[0])
    
    return None


def smallest_semi_key(vcs: "VCS") -> Optional[int]:
    smallest_carrier = smallest_semi_carrier(vcs)
    
    if smallest_carrier is None:
        return None
    
    m_semis = vcs.get_m_semis()
    for semis in m_semis.values():
        if semis.is_empty():
            continue
        
        iter_obj = semis.iterator()
        while iter_obj:
            carrier = iter_obj.carrier()
            if np.array_equal(carrier, smallest_carrier):
                return semi_key(vcs, carrier)
            iter_obj.__next__()
    
    return None


def full_exists(vcs: "VCS") -> bool:
    m_fulls = vcs.get_m_fulls()
    for fulls in m_fulls.values():
        if not fulls.is_empty():
            return True
    return False


def full_exists_at(vcs: "VCS", x: int, y: int) -> bool:
    m_fulls = vcs.get_m_fulls()
    fulls = m_fulls.get((x, y))
    return fulls is not None and not fulls.is_empty()


def semi_exists(vcs: "VCS") -> bool:
    m_semis = vcs.get_m_semis()
    for semis in m_semis.values():
        if not semis.is_empty():
            return True
    return False


def semi_exists_at(vcs: "VCS", x: int, y: int) -> bool:
    m_semis = vcs.get_m_semis()
    semis = m_semis.get((x, y))
    return semis is not None and not semis.is_empty()


def get_full_carriers(vcs: "VCS") -> CarrierList:
    result = CarrierList()
    
    m_fulls = vcs.get_m_fulls()
    for fulls in m_fulls.values():
        iter_obj = fulls.iterator()
        while iter_obj:
            result.add_new(iter_obj.carrier().copy())
            iter_obj.__next__()
    
    return result


def get_semi_carriers(vcs: "VCS") -> CarrierList:
    result = CarrierList()
    
    m_semis = vcs.get_m_semis()
    for semis in m_semis.values():
        iter_obj = semis.iterator()
        while iter_obj:
            result.add_new(iter_obj.carrier().copy())
            iter_obj.__next__()
    
    return result


def semi_intersection(vcs: "VCS") -> NDArray[np.bool_]:
    n_positions: int = vcs.get_n_positions()
    m_semis = vcs.get_m_semis()
    if not m_semis:
        return np.ones(n_positions, dtype=bool)
    
    result: Optional[NDArray[np.bool_]] = None
    
    for semis in m_semis.values():
        if semis.is_empty():
            continue
        intersection = semis.get_intersection()
        if intersection.size != n_positions:
            continue
        if result is None:
            result = intersection.copy()
        else:
            result &= intersection
    
    return result if result is not None and result.size == n_positions else np.ones(n_positions, dtype=bool)


def get_full_nbs(vcs: "VCS", x: int) -> NDArray[np.bool_]:
    n_positions: int = vcs.get_n_positions()
    result: NDArray[np.bool_] = np.zeros(n_positions, dtype=bool)
    
    m_fulls = vcs.get_m_fulls()
    for (key_x, key_y), fulls in m_fulls.items():
        if key_x == x or key_y == x:
            iter_obj = fulls.iterator()
            while iter_obj:
                result |= iter_obj.carrier()
                iter_obj.__next__()
    
    return result


def get_semi_nbs(vcs: "VCS", x: int) -> NDArray[np.bool_]:
    n_positions: int = vcs.get_n_positions()
    result: NDArray[np.bool_] = np.zeros(n_positions, dtype=bool)
    
    m_semis = vcs.get_m_semis()
    for (key_x, key_y), semis in m_semis.items():
        if key_x == x or key_y == x:
            iter_obj = semis.iterator()
            while iter_obj:
                result |= iter_obj.carrier()
                iter_obj.__next__()
    
    return result


def full_intersection(vcs: "VCS", x: int, y: int) -> NDArray[np.bool_]:
    n_positions: int = vcs.get_n_positions()
    m_fulls = vcs.get_m_fulls()
    fulls = m_fulls.get((x, y))
    if not fulls or fulls.is_empty():
        return np.ones(n_positions, dtype=bool)
    return fulls.get_intersection()


def full_greedy_union(vcs: "VCS", x: int, y: int) -> NDArray[np.bool_]:
    n_positions: int = vcs.get_n_positions()
    m_fulls = vcs.get_m_fulls()
    fulls = m_fulls.get((x, y))
    if not fulls or fulls.is_empty():
        return np.zeros(n_positions, dtype=bool)
    return fulls.get_greedy_union()


def semi_greedy_union(vcs: "VCS", x: int, y: int) -> NDArray[np.bool_]:
    n_positions: int = vcs.get_n_positions()
    m_semis = vcs.get_m_semis()
    semis = m_semis.get((x, y))
    if not semis or semis.is_empty():
        return np.zeros(n_positions, dtype=bool)
    return semis.get_greedy_union()

