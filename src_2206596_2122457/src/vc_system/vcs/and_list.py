import numpy as np
from typing import Optional
from numpy.typing import NDArray
from src_2206596_2122457.src.vc_system.carrier_list import CarrierList


class AndList(CarrierList):
    def __init__(
        self,
        initial_carrier: Optional[NDArray[np.bool_]] = None,
        carriers_list: Optional[list[NDArray[np.bool_]]] = None,
    ) -> None:
        if initial_carrier is not None:
            super().__init__()
            self.add_new(initial_carrier.copy())
        elif carriers_list is not None:
            super().__init__()
            for carrier in carriers_list:
                carrier_bool: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
                self.add_new(carrier_bool)
        else:
            super().__init__()
        
        self._processed_intersection: Optional[NDArray[np.bool_]] = None
        self._n_positions: int = (
            int(self.m_list[0].carrier.size)
            if self.m_list
            else 0
        )

    def remove_supersets_of(self, carrier: NDArray[np.bool_]) -> bool:
        q: NDArray[np.bool_] = np.asarray(carrier, dtype=bool)
        removed_any: bool = self.remove_supersets_of_check_old_removed(q)
        if removed_any:
            self.calc_intersection()
        return removed_any

    def add(self, carrier: NDArray[np.bool_]) -> None:
        self.remove_supersets_of(carrier)
        carrier_bool: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
        self.add_new(carrier_bool)

    def try_add(self, carrier: NDArray[np.bool_], limit: bool) -> bool:
        q: NDArray[np.bool_] = np.asarray(carrier, dtype=bool)
        
        if self.superset_of_any(q):
            return False
        
        if limit:
            if self.remove_supersets_of(carrier):
                carrier_copy_1: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
                self.add_new(carrier_copy_1)
                return True
            
            if self.is_empty():
                carrier_copy_2: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
                self.add_new(carrier_copy_2)
                return True
            
            current_intersection: NDArray[np.bool_] = self.get_all_intersection()
            if current_intersection.size == 0:
                carrier_copy_3: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
                self.add_new(carrier_copy_3)
                return True
            
            new_intersection: NDArray[np.bool_] = current_intersection & q
            
            if not np.array_equal(current_intersection, new_intersection):
                carrier_copy_4: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
                self.add_new(carrier_copy_4)
                return True
            return False
        
        self.add(carrier)
        return True

    def get_intersection(self) -> NDArray[np.bool_]:
        if self._processed_intersection is not None:
            return self._processed_intersection.copy()
        return self.get_all_intersection()

    def try_set_processed(self, carrier: NDArray[np.bool_]) -> bool:
        carrier_bool: NDArray[np.bool_] = np.asarray(carrier, dtype=bool).copy()
        
        if self.try_set_old(carrier_bool):
            if self._processed_intersection is None:
                self._processed_intersection = carrier_bool.copy()
            else:
                self._processed_intersection &= carrier_bool
            return True
        return False

    def mark_all_unprocessed(self) -> None:
        self.mark_all_new()
        self._processed_intersection = None

    def calc_intersection(self) -> None:
        self._processed_intersection = self.get_old_intersection()

