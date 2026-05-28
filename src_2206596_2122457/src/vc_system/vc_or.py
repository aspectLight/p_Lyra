import numpy as np
from typing import List
from src_2206596_2122457.src.vc_system.carrier_list import CarrierList


class VCOrCombiner:
    
    def __init__(
        self,
        semis: CarrierList,
        fulls: CarrierList,
        x_captured_set: np.ndarray,
        y_captured_set: np.ndarray,
    ) -> None:
        self.x_captured_set = np.asarray(x_captured_set, dtype=bool)
        self.y_captured_set = np.asarray(y_captured_set, dtype=bool)
        self.n_positions = len(self.x_captured_set)
        
        self.m_mem: List[np.ndarray] = []
        new_semis_count = 0
        old_semis_count = 0
        
        it = semis.iterator()
        while it:
            table = self._normalize_carrier(it.carrier())
            if not it.old():
                self.m_mem.append(table)
                new_semis_count += 1
            it.__next__()
        
        if new_semis_count == 0:
            return
        
        it = semis.iterator()
        while it:
            table = self._normalize_carrier(it.carrier())
            if it.old():
                self.m_mem.append(table)
                old_semis_count += 1
            it.__next__()
        
        fulls_count = 0
        it = fulls.iterator()
        while it:
            table = self._normalize_carrier(it.carrier())
            self.m_mem.append(table)
            fulls_count += 1
            it.__next__()
        
        self._search(
            forbidden=np.zeros(self.n_positions, dtype=bool),
            capture_x=True,
            capture_y=True,
            new_semis=0,
            new_semis_count=new_semis_count,
            old_semis_count=old_semis_count,
            filtered_count=fulls_count,
        )
    
    def _normalize_carrier(self, carrier: np.ndarray) -> np.ndarray:
        table = np.asarray(carrier, dtype=bool)
        if len(table) != self.n_positions:
            raise ValueError(
                f"Carrier table length {len(table)} != expected {self.n_positions}"
            )
        return np.ascontiguousarray(table)
    
    def search_result(self) -> List[np.ndarray]:
        return self.m_mem
    
    def _search(
        self,
        forbidden: np.ndarray,
        capture_x: bool,
        capture_y: bool,
        new_semis: int,
        new_semis_count: int,
        old_semis_count: int,
        filtered_count: int,
    ) -> int:
        if new_semis_count <= 0:
            raise ValueError("new_semis_count must be positive")
        
        old_semis = new_semis + new_semis_count
        
        i_new = self._intersect(new_semis, new_semis_count)
        i_old = self._intersect(old_semis, old_semis_count)
        i = i_new & i_old
        
        captured_set = np.zeros(self.n_positions, dtype=bool)
        if capture_x:
            captured_set |= self.x_captured_set
        if capture_y:
            captured_set |= self.y_captured_set
        
        if (i & ~captured_set).any():
            self.m_mem = self.m_mem[:new_semis]
            return 0
        
        filtered = old_semis + old_semis_count
        new_conn = filtered + filtered_count
        new_conn_count = 0
        
        if filtered_count == 0:
            min_captured_set = np.zeros(self.n_positions, dtype=bool)
            if (i & self.x_captured_set).any():
                min_captured_set |= self.x_captured_set
            if (i & self.y_captured_set).any():
                min_captured_set |= self.y_captured_set
            
            new_t = self._add(new_semis, new_semis_count + old_semis_count, min_captured_set)
            self.m_mem.append(new_t)
            filtered_count += 1
            new_conn_count += 1
        
        forbidden = forbidden | i_new
        
        while True:
            allowed, min_size = self._find_min_allowed(filtered, filtered_count, forbidden)
            
            if min_size == 0:
                for i_idx in range(new_conn_count):
                    self.m_mem[new_semis + i_idx] = self.m_mem[new_conn + i_idx]
                self.m_mem = self.m_mem[:new_semis + new_conn_count]
                return new_conn_count
            
            a_idx = np.flatnonzero(allowed)[0]
            forbidden[a_idx] = True
            
            rec_new_semis = filtered + filtered_count
            rec_new_semis_count = self._filter(new_semis, new_semis_count, a_idx)
            rec_old_semis_count = self._filter(old_semis, old_semis_count, a_idx)
            rec_filtered_count = self._filter(filtered, filtered_count, a_idx)
            
            rec_capture_x = capture_x and (not self.x_captured_set[a_idx])
            rec_capture_y = capture_y and (not self.y_captured_set[a_idx])
            
            rec_new_conn_count = self._search(
                forbidden=forbidden,
                capture_x=rec_capture_x,
                capture_y=rec_capture_y,
                new_semis=rec_new_semis,
                new_semis_count=rec_new_semis_count,
                old_semis_count=rec_old_semis_count,
                filtered_count=rec_filtered_count,
            )
            filtered_count += rec_new_conn_count
            new_conn_count += rec_new_conn_count
    
    def _find_min_allowed(
        self,
        filtered: int,
        filtered_count: int,
        forbidden: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        if filtered_count == 0:
            return np.zeros(self.n_positions, dtype=bool), 0
        
        min_size = np.inf
        allowed_idx = 0
        
        for i_idx in range(filtered_count):
            a = self.m_mem[filtered + i_idx] & ~forbidden
            size = a.sum()
            if size < min_size:
                min_size = size
                allowed_idx = i_idx
        
        allowed = self.m_mem[filtered + allowed_idx] & ~forbidden
        return allowed, int(min_size)
    
    def _intersect(self, start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.ones(self.n_positions, dtype=bool)
        
        if count == 1:
            return self.m_mem[start].copy()
        
        slice_data = np.array([self.m_mem[start + idx] for idx in range(count)], dtype=bool)
        return np.all(slice_data, axis=0)
    
    def _add(
        self,
        start: int,
        count: int,
        captured_set: np.ndarray,
    ) -> np.ndarray:
        u = captured_set.copy()
        i = np.ones(self.n_positions, dtype=bool)
        
        for idx in range(count):
            if idx >= len(self.m_mem) - start:
                raise ValueError(f"Index {idx} out of bounds for _add")
            
            next_table = self.m_mem[start + idx]
            
            if not (i & ~next_table).any():
                continue
            
            i &= next_table
            u |= next_table
            
            if not (i & ~captured_set).any():
                break
        
        return u
    
    def _filter(self, start: int, count: int, a: int) -> int:
        res = 0
        new_carriers: List[np.ndarray] = []
        
        for idx in range(count):
            s = self.m_mem[start + idx]
            if not s[a]:
                new_carriers.append(s.copy())
                res += 1
        
        self.m_mem.extend(new_carriers)
        return res


def vc_or(
    semis: CarrierList,
    fulls: CarrierList,
    x_captured_set: np.ndarray,
    y_captured_set: np.ndarray,
) -> List[np.ndarray]:
    x_captured_set = np.asarray(x_captured_set, dtype=bool)
    y_captured_set = np.asarray(y_captured_set, dtype=bool)
    
    if len(x_captured_set) != len(y_captured_set):
        raise ValueError("x_captured_set and y_captured_set must have same length")
    
    combiner = VCOrCombiner(semis, fulls, x_captured_set, y_captured_set)
    return combiner.search_result()