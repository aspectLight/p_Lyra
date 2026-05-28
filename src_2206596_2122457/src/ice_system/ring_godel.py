from src_2206596_2122457.pipeline.scripts.generate_ring_constants import BITS_PER_SLICE
from src_2206596_2122457.pipeline.scripts.ring_constants import EMPTY_RING_VALUE, GODEL_TO_INDEX, SLICE_MASK

class RingGodel:
    
    def __init__(self, value: int | None = None) -> None:
        if value is None:
            self.value: int = EMPTY_RING_VALUE
        else:
            self.value = value
    
    def add_color_to_slice(self, slice_index: int, color: int) -> None:
        shift: int = slice_index * BITS_PER_SLICE
        bits: int = color << shift
        self.value |= bits
    
    def remove_color_from_slice(self, slice_index: int, color: int) -> None:
        shift: int = slice_index * BITS_PER_SLICE
        bits: int = color << shift
        self.value &= ~bits
    
    def set_slice_to_color(self, slice_index: int, color: int) -> None:
        shift: int = slice_index * BITS_PER_SLICE
        self.value &= ~(SLICE_MASK << shift)
        self.value |= color << shift
    
    def set_empty(self) -> None:
        self.value = EMPTY_RING_VALUE
    
    def index(self) -> int:
        if self.value >= len(GODEL_TO_INDEX):
            return -1
        return int(GODEL_TO_INDEX[self.value])
    
    def to_int(self) -> int:
        return self.value