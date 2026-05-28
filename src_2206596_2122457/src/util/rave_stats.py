import numpy as np
from numpy.typing import NDArray

from src_2206596_2122457.src.constants import TOTAL_CELLS

class RAVEStats:
    __slots__ = ("max_actions", "visit_count", "q_value", "rave_prior_value",
                 "rave_prior_count", "_eps")

    def __init__(self, max_actions: int = TOTAL_CELLS,
                 rave_prior_value: float = 0.5,
                 rave_prior_count: int = 8):
        self.max_actions: int = max_actions
        self.rave_prior_value: float = rave_prior_value
        self.rave_prior_count: int = rave_prior_count
        self._eps: float = 1e-8

        self.visit_count: NDArray[np.int32] = np.full(
            max_actions, rave_prior_count, dtype=np.int32, order="C"
        )
        self.q_value: NDArray[np.float32] = np.full(
            max_actions, rave_prior_value, dtype=np.float32, order="C"
        )

    def update(self, actions_played: NDArray[np.int32], value: float) -> None:
        aids = np.asarray(actions_played, dtype=np.int32)
        old_counts = self.visit_count[aids]
        old_qs = self.q_value[aids]

        np.multiply(old_qs, old_counts, out=old_qs)
        np.add(old_qs, value, out=old_qs)
        np.divide(old_qs, old_counts + 1, out=old_qs)

        self.visit_count[aids] = old_counts + 1
        self.q_value[aids] = old_qs
    
    def blend_array(self, action_ids: NDArray[np.int32], q_children: NDArray[np.float32], visit_count: int) -> NDArray[np.float32]:
        amaf_counts = self.visit_count[action_ids]
        rave_qs = self.q_value[action_ids]

        beta = amaf_counts / (visit_count + amaf_counts + self._eps)
        mask = amaf_counts > self.rave_prior_count
        blended = q_children.copy()
        blended[mask] += beta[mask] * (rave_qs[mask] - q_children[mask])
        return blended
