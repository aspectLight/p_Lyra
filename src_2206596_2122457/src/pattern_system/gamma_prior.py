import numpy as np
from typing import Optional, Dict
from src_2206596_2122457.src.pattern_system.field_propagation_pattern_store import FieldPropagationPatternStore
from src_2206596_2122457.src.precomputed.global_pattern_table import PATTERN_TABLE as GLOBAL_PATTERN_TABLE # type: ignore
from src_2206596_2122457.src.precomputed.local_pattern_table import PATTERN_TABLE as LOCAL_PATTERN_TABLE  # type: ignore
from src_2206596_2122457.src.constants import TOTAL_CELLS


class GammaPriorCalculator:
    def __init__(self, store: FieldPropagationPatternStore) -> None:
        self.store: FieldPropagationPatternStore = store
        self.global_table: Dict[int, Dict[int, Dict[str, float]]] = GLOBAL_PATTERN_TABLE
        self.local_table: Dict[int, Dict[int, Dict[str, float]]] = LOCAL_PATTERN_TABLE

    def set_store(self, store: FieldPropagationPatternStore) -> None:
        self.store = store

    def compute_priors(
        self,
        player: int,
        legal_moves: np.ndarray,
        last_move: Optional[int],
        pruning_threshold: float,
    ) -> np.ndarray:
        if legal_moves.size == 0:
            return np.zeros(0, dtype=np.float32)
        priors = np.ones(int(legal_moves.size), dtype=np.float32)
        gt_player: Dict[int, Dict[str, float]] = self.global_table.get(player, {})
        get_keys = self.store.get_keys
        for i, mv in enumerate(legal_moves):
            k6, k12 = get_keys(int(mv))
            pd = gt_player.get(int(k12))
            if pd is None:
                pd = gt_player.get(int(k6))
            if pd is not None:
                priors[i] = float(pd.get("gamma", 1.0))
            else:
                priors[i] = 1.0
        if last_move is not None:
            lt_player: Dict[int, Dict[str, float]] = self.local_table.get(player, {})
            index_of = np.full(TOTAL_CELLS, -1, dtype=np.int32)
            index_of[legal_moves] = np.arange(legal_moves.size, dtype=np.int32)
            affected: np.ndarray = self.store.get_affected_neighbors(int(last_move))
            for ai in affected:
                ai_int: int = int(ai)
                idx: int = int(index_of[ai_int])
                if idx == -1:
                    continue
                k6, k12 = get_keys(ai_int)
                pd = lt_player.get(int(k12))
                if pd is None:
                    pd = lt_player.get(int(k6))
                if pd is None:
                    continue
                local_gamma = float(pd.get("gamma", 0.0))
                priors[idx] += local_gamma
        mask = priors >= float(pruning_threshold)
        if not bool(np.any(mask)):
            return np.full(int(legal_moves.size), 1.0 / float(max(1, legal_moves.size)), dtype=np.float32)
        total = float(np.sum(priors[mask], dtype=np.float64))
        out = np.zeros(int(legal_moves.size), dtype=np.float32)
        if total > 0.0:
            out[mask] = (priors[mask] / total).astype(np.float32)
        else:
            out[:] = 1.0 / float(max(1, legal_moves.size))
        return out