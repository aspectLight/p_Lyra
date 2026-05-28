from src_2206596_2122457.pipeline.scripts.generate_ring_constants import BITS_PER_SLICE
from src_2206596_2122457.pipeline.scripts.ring_constants import SLICE_MASK
from .ring_godel import RingGodel

class PatternRingGodel(RingGodel):
    
    def __init__(self, value: int | None = None) -> None:
        super().__init__(value)
        self.mask: int = 0
    
    def set_empty(self) -> None:
        super().set_empty()
        self.mask = 0
    
    def add_slice_to_mask(self, slice_index: int) -> None:
        shift: int = slice_index * BITS_PER_SLICE
        self.mask |= SLICE_MASK << shift
    
    def matches(self, other: "PatternRingGodel") -> bool:
        masked_self: int = self.value & self.mask
        masked_other: int = other.value & self.mask
        return (masked_other & masked_self) == masked_self