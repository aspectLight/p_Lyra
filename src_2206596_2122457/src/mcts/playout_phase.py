import numpy as np
from typing import List, Tuple, Optional

from src_2206596_2122457.src.pattern_system.playout_policy import PlayoutPolicy
from src_2206596_2122457.src.pattern_system.field_propagation_pattern_store import FieldPropagationPatternStore
from src_2206596_2122457.src.config import MCTSConfig
from src_2206596_2122457.src.cluster_system.clusters import Clusters
from src_2206596_2122457.src.constants import BOARD_SIZE, TOTAL_CELLS, PIECE_B, PIECE_R
from src_2206596_2122457.src.util.iteration_logger import IterationLogger
from src_2206596_2122457.src.precomputed.global_pattern_table import PATTERN_TABLE as GLOBAL_PATTERN_TABLE  # type: ignore
from src_2206596_2122457.src.precomputed.local_pattern_table import PATTERN_TABLE as LOCAL_PATTERN_TABLE  # type: ignore
from src_2206596_2122457.src.util.board_utils import board_with_pattern_table


class PlayoutPhase:
    def __init__(self, config: MCTSConfig):
        self.board_size = BOARD_SIZE
        self.use_rave: bool = config.use_rave
        self.use_pattern_playout: bool = config.use_pattern_playout

        self.store: FieldPropagationPatternStore = FieldPropagationPatternStore()
        self.root_store: FieldPropagationPatternStore = self.store
        self.playout_policy: Optional[PlayoutPolicy] = PlayoutPolicy(self.store) if self.use_pattern_playout else None

    def set_root_store_from_board(self, board_state: np.ndarray) -> None:
        self.store.reset()
        for idx, val in enumerate(board_state):
            if int(val) == PIECE_B or int(val) == PIECE_R:
                q = idx // self.board_size
                r = idx % self.board_size
                self.store.apply_move(int(q), int(r), int(val))
        self.store.initialize_all_pattern_keys()
        self.root_store = self.store
        if self.playout_policy is not None:
            self.playout_policy.store = self.root_store

    def execute_rollout(
        self,
        starting_player_piece: int,
        board_state: np.ndarray,
        empty_cells: np.ndarray,
        clusters: Clusters,
        logger: Optional[IterationLogger] = None,
    ) -> Tuple[float, List[int]]:
        local_board = board_state
        rollout_clusters = clusters.copy()
        
        if empty_cells.dtype != np.int32:
            empty_cells = empty_cells.astype(np.int32)

        if self.playout_policy is not None:
            local_store = self.root_store.copy()
            self.playout_policy.store = local_store
            self.playout_policy.initialize_for_playout(empty_cells)

        prealloc_actions = None
        action_idx = 0
        if self.use_rave:
            prealloc_actions = np.empty(int(empty_cells.size), dtype=np.int32)

        current_player = int(starting_player_piece)
        steps: int = 0
        max_steps: int = int(empty_cells.size)

        available_moves = empty_cells.copy()
        active_count: int = int(available_moves.size)
        move_to_index = np.full(TOTAL_CELLS, -1, dtype=np.int32)
        move_to_index[available_moves[:active_count]] = np.arange(active_count, dtype=np.int32)

        last_move: Optional[int] = None

        while steps < max_steps:
            if active_count == 0:
                break

            proposed = None
            if self.playout_policy is not None:
                proposed = self.playout_policy.generate_move(current_player, last_move)
                if logger is not None:
                    if proposed is not None:
                        key6, key12 = self.root_store.get_keys(int(proposed))
                        gp = GLOBAL_PATTERN_TABLE.get(current_player, {}).get(int(key12)) or GLOBAL_PATTERN_TABLE.get(current_player, {}).get(int(key6))
                        lp = None
                        if last_move is not None:
                            lp = LOCAL_PATTERN_TABLE.get(current_player, {}).get(int(key12)) or LOCAL_PATTERN_TABLE.get(current_player, {}).get(int(key6))
                        logger.write(f"ROLLOUT_PROPOSED player={current_player} move={int(proposed)} g={(gp.get('gamma') if gp else None)} l={(lp.get('gamma') if lp else None)}")

            if proposed is not None and 0 <= proposed < TOTAL_CELLS and move_to_index[proposed] != -1:
                idx = int(move_to_index[proposed])
            else:
                idx = int(np.random.randint(0, active_count))

            move = int(available_moves[idx])
            if logger is not None:
                store_for_lookup = self.playout_policy.store if self.playout_policy is not None else self.root_store
                k6, k12 = store_for_lookup.get_keys(int(move))
                gp = GLOBAL_PATTERN_TABLE.get(current_player, {}).get(int(k12)) or GLOBAL_PATTERN_TABLE.get(current_player, {}).get(int(k6))
                lp = None
                if last_move is not None:
                    lp = LOCAL_PATTERN_TABLE.get(current_player, {}).get(int(k12)) or LOCAL_PATTERN_TABLE.get(current_player, {}).get(int(k6))
                logger.write(f"=== ROLLOUT_MATCH (player={current_player}, last={last_move}, move={move}, key6={int(k6)}, key12={int(k12)}, g={gp.get('gamma') if gp else None}, l={lp.get('gamma') if lp else None}) ===")
            last_idx = active_count - 1
            last_move_val = int(available_moves[last_idx])
            available_moves[idx] = last_move_val
            move_to_index[last_move_val] = idx
            active_count -= 1
            move_to_index[move] = -1

            local_board[move] = current_player

            rollout_clusters.add_piece_and_update_groups(move, current_player)
            if logger is not None:
                logger.write(f"=== ROLLOUT_MOVE (player={current_player}, move={move}) ===")

            if self.playout_policy is not None:
                self.playout_policy.apply_move(move, current_player)
            if logger is not None:
                pattern_view = board_with_pattern_table(
                    local_board,
                    last_move,
                    self.playout_policy.store if self.playout_policy is not None else self.root_store,
                    current_player,
                    GLOBAL_PATTERN_TABLE,
                    LOCAL_PATTERN_TABLE,
                    move,
                    False,
                )
                logger.write(f"=== ROLLOUT_BOARD (player={current_player}, move={move}, last={last_move}) ===")
                logger.write_block("", pattern_view)
                logger.write("=== END ROLLOUT_BOARD ===")

            if prealloc_actions is not None:
                prealloc_actions[action_idx] = move
                action_idx += 1

            winner = rollout_clusters.get_winner()
            if winner is not None:
                value = 1.0 if winner == starting_player_piece else -1.0
                if prealloc_actions is not None:
                    return value, prealloc_actions[:action_idx].tolist()
                else:
                    return value, []

            last_move = move
            current_player = 3 - current_player
            steps += 1

        if prealloc_actions is not None:
            return 0.0, prealloc_actions[:action_idx].tolist()
        else:
            return 0.0, []