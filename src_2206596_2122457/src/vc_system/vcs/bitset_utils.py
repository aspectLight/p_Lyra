import numpy as np
from typing import List, Iterable
from numpy.typing import NDArray


def bit_for(i: int) -> int:
    return 1 << i


def bitset_from_indices(indices: Iterable[int]) -> int:
    result: int = 0
    for idx in indices:
        result |= bit_for(idx)
    return result


def indices_from_bitset(bits: int) -> List[int]:
    result: List[int] = []
    temp: int = bits
    while temp:
        lsb: int = temp & -temp
        idx: int = lsb.bit_length() - 1
        result.append(idx)
        temp ^= lsb
    return result


def bitset_intersection(a: int, b: int) -> int:
    return a & b


def bitset_union(a: int, b: int) -> int:
    return a | b


def bitset_is_subset(a: int, b: int) -> bool:
    return (a & b) == a


def bitset_is_superset(a: int, b: int) -> bool:
    return bitset_is_subset(b, a)


def bitset_from_table(table: NDArray[np.int_], value: int) -> int:
    indices: NDArray[np.int_] = np.nonzero(table == value)[0]
    return bitset_from_indices(indices)


def table_from_bitset(bits: int, n: int) -> NDArray[np.bool_]:
    result: NDArray[np.bool_] = np.zeros(n, dtype=bool)
    for idx in indices_from_bitset(bits):
        if idx < n:
            result[idx] = True
    return result


def bool_array_from_bitset(bits: int, n: int) -> NDArray[np.bool_]:
    return table_from_bitset(bits, n)


def bitset_from_bool_array(arr: NDArray[np.bool_]) -> int:
    indices: NDArray[np.int_] = np.flatnonzero(arr)
    return bitset_from_indices(indices)


def bitset_count(bits: int) -> int:
    return bin(bits).count('1')

