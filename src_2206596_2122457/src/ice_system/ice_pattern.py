from src_2206596_2122457.src.ice_system.ice_constants import PATTERN_NUM_SLICES


class IcePattern:
    __slots__ = ('index', 'type_code', 'name', 'comment', 'flags', 
                 'weight_value', 'radius', 'slices')
    
    def __init__(self, index: int) -> None:
        from src_2206596_2122457.src.precomputed.precomputed_ice_patterns import (
            PATTERNS_TYPE_CODES, PATTERNS_NAMES, PATTERNS_COMMENTS,
            PATTERNS_FLAGS, PATTERNS_WEIGHTS, PATTERNS_RADIUS, PATTERNS_SLICES
        )
        
        if not (0 <= index < len(PATTERNS_NAMES)):
            raise IndexError(f"Pattern index {index} out of range")
        
        self.index = index
        self.type_code = str(PATTERNS_TYPE_CODES[index])
        self.name = str(PATTERNS_NAMES[index])
        self.comment = str(PATTERNS_COMMENTS[index])
        self.flags = int(PATTERNS_FLAGS[index])
        self.weight_value = int(PATTERNS_WEIGHTS[index])
        self.radius = int(PATTERNS_RADIUS[index])
        self.slices = PATTERNS_SLICES[index]


class RotatedPattern:
    __slots__ = ('pattern', 'angle')
    
    def __init__(self, pattern: IcePattern, angle: int) -> None:
        if not (0 <= angle < PATTERN_NUM_SLICES):
            raise ValueError(f"Angle must be 0..{PATTERN_NUM_SLICES - 1}, got {angle}")
        self.pattern = pattern
        self.angle = angle