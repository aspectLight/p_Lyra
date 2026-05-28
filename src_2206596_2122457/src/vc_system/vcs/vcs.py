import numpy as np
from typing import Optional, Dict, Deque, Tuple
from collections import deque
from numpy.typing import NDArray
from src_2206596_2122457.src.constants import BOARD_SIZE
from src_2206596_2122457.src.group_system.groups import Groups, BLUE, RED
from src_2206596_2122457.src.vc_system.vcs.vc_builder_param import VCBuilderParam
from src_2206596_2122457.src.vc_system.vcs.value_objects import Ends, Full
from src_2206596_2122457.src.vc_system.vcs.and_list import AndList
from src_2206596_2122457.src.vc_system.vcs.or_list import OrList
from src_2206596_2122457.src.vc_system.vcs.vcs_queries import (
    get_smallest_semis_union,
    smallest_full_carrier,
    full_adjacent,
    smallest_semi_carrier,
    semi_key,
    smallest_semi_key,
    full_exists,
    full_exists_at,
    semi_exists,
    semi_exists_at,
    get_full_carriers,
    get_semi_carriers,
    semi_intersection,
    get_full_nbs,
    get_semi_nbs,
    full_intersection,
    full_greedy_union,
    semi_greedy_union,
)
from src_2206596_2122457.src.vc_system.carrier_list import CarrierList


def compute_edge_indices_for_color(color: int) -> NDArray[np.int32]:
    edge_indices: list[int] = []
    
    if color == BLUE:
        for col in range(BOARD_SIZE):
            edge_indices.append(col)
            edge_indices.append((BOARD_SIZE - 1) * BOARD_SIZE + col)
    else:
        for row in range(BOARD_SIZE):
            edge_indices.append(row * BOARD_SIZE)
            edge_indices.append(row * BOARD_SIZE + BOARD_SIZE - 1)
    
    return np.array(edge_indices, dtype=np.int32)


class VCS:
    def __init__(self, color: int, table: NDArray[np.int_]) -> None:
        self._color: int = color
        self._table: NDArray[np.int_] = np.asarray(table, dtype=np.int_)
        self._opponent_color: int = RED if color == BLUE else BLUE
        
        edge_indices: NDArray[np.int32] = compute_edge_indices_for_color(color)
        self._edge1: int = int(edge_indices[0])
        self._edge2: int = int(edge_indices[len(edge_indices) // 2])
        
        self._m_fulls: Dict[Tuple[int, int], AndList] = {}
        self._m_semis: Dict[Tuple[int, int], OrList] = {}
        
        self._m_fulls_and_queue: Deque[Full] = deque()
        self._m_semis_or_queue: Deque[Ends] = deque()
        
        self._m_param: Optional[VCBuilderParam] = None
        self._m_groups: Optional[Groups] = None
        
        self._m_captured_set: Dict[int, NDArray[np.bool_]] = {}
        self._n_positions: int = len(table)

    def get_or_create_andlist(self, x: int, y: int) -> AndList:
        key: Tuple[int, int] = (x, y)
        if key not in self._m_fulls:
            self._m_fulls[key] = AndList()
        return self._m_fulls[key]

    def get_or_create_orlist(self, x: int, y: int) -> OrList:
        key: Tuple[int, int] = (x, y)
        if key not in self._m_semis:
            self._m_semis[key] = OrList()
        return self._m_semis[key]

    def get_full_carriers_at(self, x: int, y: int) -> AndList:
        return self.get_or_create_andlist(x, y)

    def get_semi_carriers_at(self, x: int, y: int) -> OrList:
        return self.get_or_create_orlist(x, y)

    def get_color(self) -> int:
        return self._color

    def get_opponent_color(self) -> int:
        return self._opponent_color

    def get_table(self) -> NDArray[np.int_]:
        return self._table

    def get_m_param(self) -> Optional[VCBuilderParam]:
        return self._m_param

    def get_m_groups(self) -> Optional[Groups]:
        return self._m_groups

    def get_m_captured_set(self) -> Dict[int, NDArray[np.bool_]]:
        return self._m_captured_set

    def get_n_positions(self) -> int:
        return self._n_positions

    def get_m_fulls_and_queue(self) -> Deque[Full]:
        return self._m_fulls_and_queue

    def get_m_semis_or_queue(self) -> Deque[Ends]:
        return self._m_semis_or_queue

    def get_m_fulls(self) -> Dict[Tuple[int, int], AndList]:
        return self._m_fulls

    def get_m_semis(self) -> Dict[Tuple[int, int], OrList]:
        return self._m_semis

    def set_table(self, table: NDArray[np.int_]) -> None:
        self._table = np.asarray(table, dtype=np.int_)

    def set_m_param(self, param: VCBuilderParam) -> None:
        self._m_param = param

    def set_m_groups(self, groups: Groups) -> None:
        self._m_groups = groups

    def reset(self) -> None:
        self._m_fulls.clear()
        self._m_semis.clear()
        self._m_fulls_and_queue.clear()
        self._m_semis_or_queue.clear()
        self._m_captured_set.clear()

    def build(
        self,
        param: VCBuilderParam,
        groups: Groups,
    ) -> None:
        from src_2206596_2122457.src.vc_system.vcs.vcs_build import build as build_vcs
        build_vcs(self, param, groups, self._table)

    def get_smallest_semis_union(self) -> NDArray[np.bool_]:
        return get_smallest_semis_union(self)

    def smallest_full_carrier(self) -> Optional[NDArray[np.bool_]]:
        return smallest_full_carrier(self)

    def full_adjacent(self, x: int, y: int) -> int:
        return full_adjacent(self, x, y)

    def smallest_semi_carrier(self) -> Optional[NDArray[np.bool_]]:
        return smallest_semi_carrier(self)

    def semi_key(self, carrier: NDArray[np.bool_]) -> Optional[int]:
        return semi_key(self, carrier)

    def smallest_semi_key(self) -> Optional[int]:
        return smallest_semi_key(self)

    def full_exists(self) -> bool:
        return full_exists(self)

    def full_exists_at(self, x: int, y: int) -> bool:
        return full_exists_at(self, x, y)

    def semi_exists(self) -> bool:
        return semi_exists(self)

    def semi_exists_at(self, x: int, y: int) -> bool:
        return semi_exists_at(self, x, y)

    def get_full_carriers(self) -> CarrierList:
        return get_full_carriers(self)

    def get_semi_carriers(self) -> CarrierList:
        return get_semi_carriers(self)

    def semi_intersection(self) -> NDArray[np.bool_]:
        return semi_intersection(self)

    def get_full_carriers_at_public(self, x: int, y: int) -> AndList:
        return self.get_full_carriers_at(x, y)

    def get_semi_carriers_at_public(self, x: int, y: int) -> OrList:
        return self.get_semi_carriers_at(x, y)

    def get_full_nbs(self, x: int) -> NDArray[np.bool_]:
        return get_full_nbs(self, x)

    def get_semi_nbs(self, x: int) -> NDArray[np.bool_]:
        return get_semi_nbs(self, x)

    def full_intersection(self, x: int, y: int) -> NDArray[np.bool_]:
        return full_intersection(self, x, y)

    def full_greedy_union(self, x: int, y: int) -> NDArray[np.bool_]:
        return full_greedy_union(self, x, y)

    def semi_greedy_union(self, x: int, y: int) -> NDArray[np.bool_]:
        return semi_greedy_union(self, x, y)

