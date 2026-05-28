import numpy as np
from typing import Dict, Optional, Deque, Tuple
from collections import deque
from numpy.typing import NDArray
from src_2206596_2122457.src.constants import BOARD_SIZE
from src_2206596_2122457.src.group_system.groups import Groups, BLUE, RED
from src_2206596_2122457.src.vc_system.vcs.vc_builder_param import VCBuilderParam
from src_2206596_2122457.src.vc_system.vcs.value_objects import Ends, Full
from src_2206596_2122457.src.vc_system.vcs.and_list import AndList
from src_2206596_2122457.src.vc_system.vcs.or_list import OrList


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


class VCSCore:
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

    def _get_or_create_andlist(self, x: int, y: int) -> AndList:
        key: Tuple[int, int] = (x, y)
        if key not in self._m_fulls:
            self._m_fulls[key] = AndList()
        return self._m_fulls[key]

    def _get_or_create_orlist(self, x: int, y: int) -> OrList:
        key: Tuple[int, int] = (x, y)
        if key not in self._m_semis:
            self._m_semis[key] = OrList()
        return self._m_semis[key]

    def get_full_carriers_at(self, x: int, y: int) -> AndList:
        return self._get_or_create_andlist(x, y)

    def get_semi_carriers_at(self, x: int, y: int) -> OrList:
        return self._get_or_create_orlist(x, y)

    def reset(self) -> None:
        self._m_fulls.clear()
        self._m_semis.clear()
        self._m_fulls_and_queue.clear()
        self._m_semis_or_queue.clear()
        self._m_captured_set.clear()

