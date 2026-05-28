import numpy as np
from src_2206596_2122457.src.ice_system.ice_pattern import IcePattern
from src_2206596_2122457.src.ice_system.hashed_pattern_set import HashedPatternSet
from src_2206596_2122457.src.ice_system.ice_constants import CELL_BLUE
from src_2206596_2122457.src.precomputed.precomputed_ice_patterns import (
    PATTERNS_E_FILLIN, PATTERNS_FILLIN_BLUE, PATTERNS_FILLIN_RED,
    PATTERNS_S_REVERSIBLE_BLUE, PATTERNS_S_REVERSIBLE_RED,
    PATTERNS_T_REVERSIBLE_BLUE, PATTERNS_T_REVERSIBLE_RED,
    PATTERNS_INFERIOR_BLUE, PATTERNS_INFERIOR_RED,
    PATTERNS_CAPTURED_BLUE, PATTERNS_CAPTURED_RED,
    PATTERNS_VULNERABLE_BLUE, PATTERNS_VULNERABLE_RED,
    PATTERNS_REVERSIBLE_BLUE, PATTERNS_REVERSIBLE_RED
)

class IcePatternSet:
    
    def __init__(self) -> None:
        self.e_fillin = self._load_category_patterns(PATTERNS_E_FILLIN)
        self.fillin_blue = self._load_category_patterns(PATTERNS_FILLIN_BLUE)
        self.fillin_red = self._load_category_patterns(PATTERNS_FILLIN_RED)
        self.s_reversible_blue = self._load_category_patterns(PATTERNS_S_REVERSIBLE_BLUE)
        self.s_reversible_red = self._load_category_patterns(PATTERNS_S_REVERSIBLE_RED)
        self.t_reversible_blue = self._load_category_patterns(PATTERNS_T_REVERSIBLE_BLUE)
        self.t_reversible_red = self._load_category_patterns(PATTERNS_T_REVERSIBLE_RED)
        self.inferior_blue = self._load_category_patterns(PATTERNS_INFERIOR_BLUE)
        self.inferior_red = self._load_category_patterns(PATTERNS_INFERIOR_RED)
        self.captured_blue = self._load_category_patterns(PATTERNS_CAPTURED_BLUE)
        self.captured_red = self._load_category_patterns(PATTERNS_CAPTURED_RED)
        self.vulnerable_blue = self._load_category_patterns(PATTERNS_VULNERABLE_BLUE)
        self.vulnerable_red = self._load_category_patterns(PATTERNS_VULNERABLE_RED)
        self.reversible_blue = self._load_category_patterns(PATTERNS_REVERSIBLE_BLUE)
        self.reversible_red = self._load_category_patterns(PATTERNS_REVERSIBLE_RED)
        
        self._hashed_e_fillin = HashedPatternSet('e_fillin')
        self._hashed_fillin_blue = HashedPatternSet('fillin_blue')
        self._hashed_fillin_red = HashedPatternSet('fillin_red')
        self._hashed_s_reversible_blue = HashedPatternSet('s_reversible_blue')
        self._hashed_s_reversible_red = HashedPatternSet('s_reversible_red')
        self._hashed_t_reversible_blue = HashedPatternSet('t_reversible_blue')
        self._hashed_t_reversible_red = HashedPatternSet('t_reversible_red')
        self._hashed_inferior_blue = HashedPatternSet('inferior_blue')
        self._hashed_inferior_red = HashedPatternSet('inferior_red')
        self._hashed_captured_blue = HashedPatternSet('captured_blue')
        self._hashed_captured_red = HashedPatternSet('captured_red')
        self._hashed_vulnerable_blue = HashedPatternSet('vulnerable_blue')
        self._hashed_vulnerable_red = HashedPatternSet('vulnerable_red')
        self._hashed_reversible_blue = HashedPatternSet('reversible_blue')
        self._hashed_reversible_red = HashedPatternSet('reversible_red')
        
    def _load_category_patterns(self, pattern_indices: np.ndarray) -> np.ndarray:
        patterns = np.array([IcePattern(int(idx)) for idx in pattern_indices], dtype=object)
        return patterns
    
    @property
    def hashed_e_fillin(self) -> HashedPatternSet:
        return self._hashed_e_fillin
    
    @property
    def hashed_fillin_blue(self) -> HashedPatternSet:
        return self._hashed_fillin_blue
    
    @property
    def hashed_fillin_red(self) -> HashedPatternSet:
        return self._hashed_fillin_red
    
    @property
    def hashed_s_reversible_blue(self) -> HashedPatternSet:
        return self._hashed_s_reversible_blue
    
    @property
    def hashed_s_reversible_red(self) -> HashedPatternSet:
        return self._hashed_s_reversible_red
    
    @property
    def hashed_t_reversible_blue(self) -> HashedPatternSet:
        return self._hashed_t_reversible_blue
    
    @property
    def hashed_t_reversible_red(self) -> HashedPatternSet:
        return self._hashed_t_reversible_red
    
    @property
    def hashed_inferior_blue(self) -> HashedPatternSet:
        return self._hashed_inferior_blue
    
    @property
    def hashed_inferior_red(self) -> HashedPatternSet:
        return self._hashed_inferior_red
    
    @property
    def hashed_captured_blue(self) -> HashedPatternSet:
        return self._hashed_captured_blue
    
    @property
    def hashed_captured_red(self) -> HashedPatternSet:
        return self._hashed_captured_red
    
    @property
    def hashed_vulnerable_blue(self) -> HashedPatternSet:
        return self._hashed_vulnerable_blue
    
    @property
    def hashed_vulnerable_red(self) -> HashedPatternSet:
        return self._hashed_vulnerable_red
    
    @property
    def hashed_reversible_blue(self) -> HashedPatternSet:
        return self._hashed_reversible_blue
    
    @property
    def hashed_reversible_red(self) -> HashedPatternSet:
        return self._hashed_reversible_red
    
    def hashed_fillin(self, color: int) -> HashedPatternSet:
        if color == CELL_BLUE:
            return self._hashed_fillin_blue
        return self._hashed_fillin_red
    
    def hashed_s_reversible(self, color: int) -> HashedPatternSet:
        if color == CELL_BLUE:
            return self._hashed_s_reversible_blue
        return self._hashed_s_reversible_red
    
    def hashed_t_reversible(self, color: int) -> HashedPatternSet:
        if color == CELL_BLUE:
            return self._hashed_t_reversible_blue
        return self._hashed_t_reversible_red
    
    def hashed_inferior(self, color: int) -> HashedPatternSet:
        if color == CELL_BLUE:
            return self._hashed_inferior_blue
        return self._hashed_inferior_red
    
    def hashed_captured(self, color: int) -> HashedPatternSet:
        if color == CELL_BLUE:
            return self._hashed_captured_blue
        return self._hashed_captured_red
    
    def hashed_vulnerable(self, color: int) -> HashedPatternSet:
        if color == CELL_BLUE:
            return self._hashed_vulnerable_blue
        return self._hashed_vulnerable_red
    
    def hashed_reversible(self, color: int) -> HashedPatternSet:
        if color == CELL_BLUE:
            return self._hashed_reversible_blue
        return self._hashed_reversible_red