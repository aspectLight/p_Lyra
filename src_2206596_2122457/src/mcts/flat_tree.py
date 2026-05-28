import numpy as np
from typing import Dict, Optional
from src_2206596_2122457.src.constants import TOTAL_CELLS


class FlatTree:
    def __init__(self, use_rave: bool = True, rave_prior_count: int = 8, rave_prior_value: float = 0.5) -> None:
        self.use_rave: bool = use_rave
        self.rave_prior_count = rave_prior_count
        self.rave_prior_value = rave_prior_value
        self._capacity: int = 1024
        self._next_node_index: int = 0

        self.z_hash: np.ndarray = np.zeros(self._capacity, dtype=np.uint64)
        self.player_piece: np.ndarray = np.zeros(self._capacity, dtype=np.int8)
        shape = (self._capacity, TOTAL_CELLS)
        self.visit_count: np.ndarray = np.zeros(shape, dtype=np.int32)
        self.q_value: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.priors: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.children: np.ndarray = np.full(shape, -1, dtype=np.int32)
        self.parent_visit_sum: np.ndarray = np.zeros(self._capacity, dtype=np.int32)
        self.priors_initialized: np.ndarray = np.zeros(self._capacity, dtype=np.bool_)

        self.knowledge_computed: np.ndarray = np.zeros(self._capacity, dtype=np.bool_)
        self.knowledge_mask: np.ndarray = np.ones((self._capacity, TOTAL_CELLS), dtype=np.bool_)

        self.rave_visit: Optional[np.ndarray] = None
        self.rave_q: Optional[np.ndarray] = None
        if self.use_rave:
            self.rave_visit = np.full(shape, self.rave_prior_count, dtype=np.int32)
            self.rave_q = np.full(shape, self.rave_prior_value, dtype=np.float32)

        self._hash_to_index: Dict[int, int] = {}
        self.root_index: Optional[int] = None

    def _ensure_capacity(self, min_capacity: int) -> None:
        if min_capacity <= self._capacity:
            return
        new_cap = max(self._capacity * 2, min_capacity)
        def grow(arr: np.ndarray, fill: Optional[float] = None) -> np.ndarray:
            new = np.empty((new_cap, arr.shape[1]), dtype=arr.dtype)
            new[: self._capacity] = arr
            if fill is not None:
                new[self._capacity :] = fill
            else:
                new[self._capacity :] = 0
            return new
        
        def grow_1d(arr: np.ndarray, fill: Optional[float] = None) -> np.ndarray:
            return np.pad(arr, (0, new_cap - self._capacity))
        
        self.z_hash = grow_1d(self.z_hash)
        self.player_piece = grow_1d(self.player_piece)
        self.visit_count = grow(self.visit_count, 0)
        self.q_value = grow(self.q_value, 0.0)
        self.priors = grow(self.priors, 0.0)
        self.children = grow(self.children, -1)
        self.parent_visit_sum = grow_1d(self.parent_visit_sum)
        self.priors_initialized = grow_1d(self.priors_initialized)
        self.knowledge_computed = grow_1d(self.knowledge_computed)
        
        new_knowledge_mask = np.ones((new_cap, TOTAL_CELLS), dtype=np.bool_)
        new_knowledge_mask[: self._capacity] = self.knowledge_mask
        self.knowledge_mask = new_knowledge_mask
        
        if self.use_rave and self.rave_visit is not None and self.rave_q is not None:
            self.rave_visit = grow(self.rave_visit, self.rave_prior_count)
            self.rave_q = grow(self.rave_q, self.rave_prior_value)
        self._capacity = new_cap

    def _allocate_node(self, z_hash: int, player_piece: int) -> int:
        idx = self._next_node_index
        self._ensure_capacity(idx + 1)
        self.z_hash[idx] = z_hash
        self.player_piece[idx] = player_piece
        self.children[idx, :] = -1
        self.parent_visit_sum[idx] = 0
        self.priors_initialized[idx] = False
        self.knowledge_computed[idx] = False
        self.knowledge_mask[idx, :] = True
        self._hash_to_index[int(np.uint64(z_hash))] = idx
        self._next_node_index += 1
        return idx

    def create_root(self, z_hash: int, player_piece: int) -> int:
        self._hash_to_index.clear()
        self._next_node_index = 0
        root_idx = self._allocate_node(z_hash, player_piece)
        self.root_index = root_idx
        return root_idx

    def find_node(self, z_hash: int) -> Optional[int]:
        return self._hash_to_index.get(int(z_hash))

    def get_child(self, node_idx: int, action_id: int) -> int:
        return int(self.children[node_idx, action_id])

    def ensure_child(self, node_idx: int, action_id: int, next_hash: int, next_player_piece: int) -> int:
        child = int(self.children[node_idx, action_id])
        if child != -1:
            return child
        new_idx = self._allocate_node(next_hash, next_player_piece)
        self.children[node_idx, action_id] = new_idx
        if self.use_rave and self.rave_visit is not None and self.rave_q is not None:
            self.rave_visit[new_idx, :] = self.rave_prior_count
            self.rave_q[new_idx, :] = self.rave_prior_value
        return new_idx

    def get_parent_visit_sum(self, node_idx: int) -> int:
        return int(self.parent_visit_sum[node_idx])