from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class Ends:
    x: int
    y: int


@dataclass
class Full:
    x: int
    y: int
    carrier: NDArray[np.bool_]


@dataclass
class Semi:
    x: int
    y: int
    carrier: NDArray[np.bool_]
    key: int

