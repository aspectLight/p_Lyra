import numpy as np
from typing import Optional
from src_2206596_2122457.src.config import MCTSConfig


class SelectionPhase:
    def __init__(self, config: MCTSConfig):
        self.exploration_constant: float = config.exploration_constant
        self.rave_weight_final: float = config.rave_weight_final
        self.rave_prior_count: int = config.rave_prior_count
        self.rave_skip_freq: int = config.rave_randomize_freq
        self._rave_skip_counter: int = self.rave_skip_freq
        self.progressive_bias_constant: float = config.progressive_bias_constant if config.use_pattern_priors else 0.0

    def select_best_action(
        self,
        child_visits: np.ndarray,
        q_values: np.ndarray,
        priors: np.ndarray,
        parent_visits_sum: int,
        possible_actions: np.ndarray,
        rave_q: Optional[np.ndarray] = None,
        rave_counts: Optional[np.ndarray] = None,
        rave_skip: bool = False,
    ) -> int:
        if possible_actions.size == 0:
            raise ValueError("No possible actions provided")

        child_visits = np.asarray(child_visits[possible_actions], dtype=np.int32)
        q_values = np.asarray(q_values[possible_actions], dtype=np.float64)
        priors = np.asarray(priors[possible_actions], dtype=np.float64)
        uct_values = q_values.copy()
        parent_visits_sqrt = float(np.sqrt(parent_visits_sum))

        final_values: np.ndarray = q_values.copy()
        use_blend = (
            rave_q is not None and rave_counts is not None and not rave_skip
        )
        if use_blend:
            assert rave_q is not None
            assert rave_counts is not None
            rave_q_local = np.asarray(rave_q[possible_actions], dtype=np.float64)
            rave_counts_local = np.asarray(rave_counts[possible_actions], dtype=np.float64)
            n = child_visits.astype(np.float64)
            n_rave = rave_counts_local
            k_amaf = self.rave_weight_final
            w_rave = n_rave / (n + n_rave + n * n_rave / k_amaf + 1e-8)
            final_values = w_rave * rave_q_local + (1.0 - w_rave) * uct_values
        else:
            final_values = uct_values

        exploration = self.exploration_constant * priors * parent_visits_sqrt / (1.0 + child_visits)
        progressive_bias = self.progressive_bias_constant * priors / (np.sqrt(child_visits.astype(np.float64)) + 1.0)
        selection_values = final_values + exploration + progressive_bias

        max_val = float(np.max(selection_values))
        best_mask = (selection_values == max_val)
        if int(np.count_nonzero(best_mask)) == 1:
            chosen_idx = int(np.argmax(selection_values))
        else:
            best_indices = np.flatnonzero(best_mask)
            chosen_idx = int(np.random.randint(0, best_indices.size))
            chosen_idx = int(best_indices[chosen_idx])

        return int(possible_actions[chosen_idx])
    
    def update_rave_skip_counter(self) -> bool:
        # Called per search step
        self._rave_skip_counter -= 1
        if self._rave_skip_counter <= 0:
            self._rave_skip_counter = self.rave_skip_freq
            return True  # skip rave (do pure UCT this selection)
        return False
