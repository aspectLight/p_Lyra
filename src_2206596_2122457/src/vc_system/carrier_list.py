from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Elem:
    carrier: np.ndarray
    old: bool


class CarrierList:
    @dataclass
    class Iterator:
        _lst: list[Elem]
        _index: int = 0

        def carrier(self) -> np.ndarray:
            return self._lst[self._index].carrier

        def old(self) -> bool:
            return self._lst[self._index].old

        def __next__(self) -> CarrierList.Iterator:
            if self._index >= len(self._lst):
                raise StopIteration
            self._index += 1
            return self

        def __iter__(self) -> CarrierList.Iterator:
            return self

        def __bool__(self) -> bool:
            return self._index < len(self._lst)

    def __init__(self) -> None:
        self.m_list: list[Elem] = []

    @classmethod
    def from_single(cls, carrier: np.ndarray) -> CarrierList:
        instance = cls()
        carrier_bool: np.ndarray = np.asarray(carrier, dtype=bool).copy()
        instance.add_new(carrier_bool)
        return instance

    @classmethod
    def from_carriers(cls, carriers_list: list[np.ndarray]) -> CarrierList:
        instance = cls()
        for carrier in carriers_list:
            carrier_bool: np.ndarray = np.asarray(carrier, dtype=bool).copy()
            instance.m_list.append(Elem(carrier=carrier_bool, old=False))
        return instance

    def count(self) -> int:
        return len(self.m_list)

    def is_empty(self) -> bool:
        return not self.m_list

    def iterator(self) -> CarrierList.Iterator:
        return CarrierList.Iterator(_lst=self.m_list, _index=0)

    def get_greedy_union(self) -> np.ndarray:
        n: int = self.m_list[0].carrier.size if self.m_list else 0
        result: np.ndarray = np.zeros(n, dtype=bool)
        for elem in self.m_list:
            np.logical_or(result, elem.carrier, out=result)
        return result

    def superset_of_any(self, carrier: np.ndarray) -> bool:
        q: np.ndarray = np.asarray(carrier, dtype=bool)
        q_sparse: bool = q.sum() << 5 < q.size

        if q_sparse:
            q_indices: np.ndarray = np.nonzero(q)[0]
            for elem in self.m_list:
                if elem.carrier[q_indices].all():
                    return True
        else:
            for elem in self.m_list:
                if not np.any(q & ~elem.carrier):
                    return True
        return False

    def remove_supersets_of_check_any_removed(self, filter_table: CarrierList) -> bool:
        removed_any: bool = False
        it: CarrierList.Iterator = filter_table.iterator()
        while it:
            if self._remove_supersets_of(it.carrier(), check_old=False):
                removed_any = True
            it.__next__()
        return removed_any

    def get_all_intersection(self) -> np.ndarray:
        if not self.m_list:
            n: int = 0
            return np.zeros(n, dtype=bool)
        result: np.ndarray = self.m_list[0].carrier.copy()
        for elem in self.m_list[1:]:
            np.logical_and(result, elem.carrier, out=result)
        return result

    def get_old_intersection(self) -> np.ndarray:
        selected: list[np.ndarray] = [elem.carrier for elem in self.m_list if elem.old]
        if not selected:
            n: int = self.m_list[0].carrier.size if self.m_list else 0
            return np.zeros(n, dtype=bool)
        result: np.ndarray = selected[0].copy()
        for carrier in selected[1:]:
            np.logical_and(result, carrier, out=result)
        return result

    def get_new_intersection(self) -> np.ndarray:
        selected: list[np.ndarray] = [elem.carrier for elem in self.m_list if not elem.old]
        if not selected:
            n: int = self.m_list[0].carrier.size if self.m_list else 0
            return np.zeros(n, dtype=bool)
        result: np.ndarray = selected[0].copy()
        for carrier in selected[1:]:
            np.logical_and(result, carrier, out=result)
        return result

    def remove_all_containing(self, set_mask: np.ndarray) -> int:
        q: np.ndarray = np.asarray(set_mask, dtype=bool)
        return self._remove_all_containing_(set_mask=q, store_removed=False, removed=None)

    def remove_all_containing_with_removed(self, set_mask: np.ndarray, removed: list[np.ndarray]) -> int:
        q: np.ndarray = np.asarray(set_mask, dtype=bool)
        return self._remove_all_containing_(set_mask=q, store_removed=True, removed=removed)

    def _remove_all_containing_(self, set_mask: np.ndarray, store_removed: bool, removed: list[np.ndarray] | None) -> int:
        q: np.ndarray = np.asarray(set_mask, dtype=bool)
        q_sparse: bool = q.sum() << 5 < q.size
        count: int = 0
        new_list: list[Elem] = []

        if q_sparse:
            q_indices: np.ndarray = np.nonzero(q)[0]
            for elem in self.m_list:
                if elem.carrier[q_indices].all():
                    if store_removed and removed is not None:
                        removed.append(elem.carrier.copy())
                    count += 1
                else:
                    new_list.append(elem)
        else:
            for elem in self.m_list:
                if np.any(q & ~elem.carrier):
                    new_list.append(elem)
                else:
                    if store_removed and removed is not None:
                        removed.append(elem.carrier.copy())
                    count += 1

        self.m_list = new_list
        return count

    def remove_supersets_of_unchecked(self, carrier: np.ndarray) -> None:
        self._remove_supersets_of(carrier, check_old=True)

    def remove_supersets_of_check_old_removed(self, carrier: np.ndarray) -> bool:
        return self._remove_supersets_of(carrier, check_old=True)

    def remove_supersets_of_check_any_removed_carrier(self, carrier: np.ndarray) -> bool:
        return self._remove_supersets_of(carrier, check_old=False)

    def _remove_supersets_of(self, carrier: np.ndarray, check_old: bool) -> bool:
        q: np.ndarray = np.asarray(carrier, dtype=bool)
        q_sparse: bool = q.sum() << 5 < q.size
        removed_any: bool = False
        new_list: list[Elem] = []

        if q_sparse:
            q_indices: np.ndarray = np.nonzero(q)[0]
            for elem in self.m_list:
                if check_old and not elem.old:
                    new_list.append(elem)
                elif not elem.carrier[q_indices].all():
                    new_list.append(elem)
                else:
                    removed_any = True
        else:
            for elem in self.m_list:
                if check_old and not elem.old:
                    new_list.append(elem)
                elif np.any(q & ~elem.carrier):
                    new_list.append(elem)
                else:
                    removed_any = True

        self.m_list = new_list
        return removed_any

    def add_new(self, carrier: np.ndarray) -> None:
        q: np.ndarray = np.asarray(carrier, dtype=bool).copy()
        for elem in self.m_list:
            if np.array_equal(elem.carrier, q):
                return
        self.m_list.append(Elem(carrier=q, old=False))

    def try_set_old(self, carrier: np.ndarray) -> bool:
        q: np.ndarray = np.asarray(carrier, dtype=bool)
        for elem in self.m_list:
            if np.array_equal(elem.carrier, q):
                if not elem.old:
                    elem.old = True
                    return True
                return False
        return False

    def mark_all_old(self) -> None:
        for elem in self.m_list:
            elem.old = True

    def mark_all_new(self) -> None:
        for elem in self.m_list:
            elem.old = False

    def clear(self) -> None:
        self.m_list = []