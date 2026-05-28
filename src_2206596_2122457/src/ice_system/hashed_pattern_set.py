import numpy as np
from src_2206596_2122457.src.ice_system.ring_godel import RingGodel
from src_2206596_2122457.src.precomputed import precomputed_ice_patterns 

class HashedPatternSet:
    
    def __init__(self, category_name: str) -> None:
        var_name = f"GODEL_MAP_{category_name.upper()}"
        if not hasattr(precomputed_ice_patterns, var_name):
            raise ValueError(f"Unknown category: {category_name}")
        
        self.category_name = category_name
        self.godel_list = getattr(precomputed_ice_patterns, var_name)
    
    def list_for_godel(self, godel: RingGodel) -> np.ndarray:
        godel_idx = godel.index()
        
        if godel_idx < 0 or godel_idx >= len(self.godel_list):
            return np.array([], dtype=object)
        
        patterns = self.godel_list[godel_idx]
        
        if not patterns:
            return np.array([], dtype=object)
        
        return np.array(patterns, dtype=object)