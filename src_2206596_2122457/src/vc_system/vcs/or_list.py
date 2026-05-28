import numpy as np
from typing import Optional
from numpy.typing import NDArray
from src_2206596_2122457.src.vc_system.carrier_list import CarrierList


class OrList(CarrierList):
    def __init__(
        self,
        carrier: Optional[NDArray[np.bool_]] = None,
        carrier_list: Optional[CarrierList] = None,
        intersection: Optional[NDArray[np.bool_]] = None,
    ) -> None:
        if carrier is not None:
            super().__init__()
            carrier_bool: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
            self.add_new(carrier_bool)
            self._intersection: NDArray[np.bool_] = carrier_bool.copy()
        elif carrier_list is not None:
            super().__init__()
            it = carrier_list.iterator()
            while it:
                self.add_new(it.carrier().copy())
                it.__next__()
            self._intersection = (
                intersection.copy()
                if intersection is not None
                else self.get_all_intersection()
            )
        else:
            super().__init__()
            self._intersection = np.array([], dtype=bool)
        
        self._queued: bool = False
        self._n_positions: int = (
            int(self.m_list[0].carrier.size)
            if self.m_list
            else 0
        )

    def try_add(self, carrier: NDArray[np.bool_]) -> bool:
        q: NDArray[np.bool_] = np.asarray(carrier, dtype=bool)
        
        if self.superset_of_any(q):
            return False
        
        self.remove_supersets_of_unchecked(q)
        carrier_bool: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
        self.add_new(carrier_bool)
        self._intersection &= carrier_bool
        return True

    def try_add_with_filter(self, carrier: NDArray[np.bool_], filter_list: CarrierList) -> bool:
        q: NDArray[np.bool_] = np.asarray(carrier, dtype=bool)
        
        if filter_list.superset_of_any(q):
            return False
        
        return self.try_add(carrier)

    def get_intersection(self) -> NDArray[np.bool_]:
        if self.is_empty() or self._intersection.size == 0:
            if self._n_positions > 0:
                return np.ones(self._n_positions, dtype=bool)
            if self.m_list:
                carrier_size: int = int(self.m_list[0].carrier.size)
                return np.ones(carrier_size, dtype=bool)
            return np.array([], dtype=bool)
        return self._intersection.copy()

    def remove_supersets_of(self, carrier: NDArray[np.bool_]) -> bool:
        q: NDArray[np.bool_] = np.asarray(carrier, dtype=bool)
        removed_any: bool = self.remove_supersets_of_check_any_removed_carrier(q)
        if removed_any:
            self.calc_intersection()
        return removed_any

    def remove_supersets_of_filter(self, filter_list: CarrierList) -> bool:
        removed_any: bool = self.remove_supersets_of_check_any_removed(filter_list)
        if removed_any:
            self.calc_intersection()
        return removed_any

    def try_queue(self, captured_set: NDArray[np.bool_]) -> bool:
        prev_queued: bool = self._queued
        captured: NDArray[np.bool_] = np.asarray(captured_set, dtype=bool)
        
        if captured.size != self._intersection.size:
            return False
        
        is_subset_result: bool = bool(np.all(self._intersection <= captured))
        self._queued = is_subset_result
        return not prev_queued and self._queued

    def mark_all_processed(self) -> None:
        self.mark_all_old()
        self._queued = False

    def mark_all_unprocessed(self) -> None:
        self.mark_all_new()

    def calc_intersection(self) -> None:
        if self.is_empty():
            if self._n_positions > 0:
                self._intersection = np.ones(self._n_positions, dtype=bool)
            else:
                self._intersection = np.array([], dtype=bool)
        else:
            self._intersection = self.get_all_intersection()

    def clear(self) -> None:
        super().clear()
        if self._n_positions > 0:
            self._intersection = np.ones(self._n_positions, dtype=bool)
        else:
            self._intersection = np.array([], dtype=bool)

