import numpy as np
from typing import List, Optional
from numpy.typing import NDArray

from src_2206596_2122457.src.cluster_system.clusters import Clusters
from src_2206596_2122457.src.mcts.selection_phase import SelectionPhase
from src_2206596_2122457.src.mcts.playout_phase import PlayoutPhase
from src_2206596_2122457.src.constants import TOTAL_CELLS
from src_2206596_2122457.src.game_trackers.inc_empty_cells_tracker import IncrementalEmptyCellsTracker
from src_2206596_2122457.src.mcts.flat_tree import FlatTree
from src_2206596_2122457.src.util.inc_zobrist_hasher_tracker import Zobrist
from src_2206596_2122457.src.config import MCTSConfig
from src_2206596_2122457.src.util.iteration_logger import IterationLogger
from src_2206596_2122457.src.util.board_utils import board_to_hex_ascii
from src_2206596_2122457.src.pattern_system.gamma_prior import GammaPriorCalculator
from src_2206596_2122457.src.ice_system.ice_engine.ice_engine import ICEngine
from src_2206596_2122457.src.board.board import Board
from src_2206596_2122457.src.group_system.groups import GroupBuilder
from src_2206596_2122457.src.ice_system.pattern_state import PatternState

PROVEN_LOSS: float = -1.0


class SearchExecutor:
    def __init__(
        self,
        selection_phase: SelectionPhase,
        playout_phase: PlayoutPhase,
        ice_engine: Optional[ICEngine] = None,
    ) -> None:
        self.selection_phase: SelectionPhase = selection_phase
        self.playout_phase: PlayoutPhase = playout_phase
        self.ice_engine: Optional[ICEngine] = ice_engine
        self._gamma_calculator: Optional[GammaPriorCalculator] = None
        self.ice_board: Optional[Board] = None
        self._config: Optional[MCTSConfig] = None

    def run_search_iterations(
        self,
        flat_tree: FlatTree,
        root_index: int,
        num_iterations: int,
        starting_player: int,
        board_state: NDArray[np.int_],
        empty_tracker: IncrementalEmptyCellsTracker,
        clusters: Clusters,
        zobrist: Zobrist,
        last_move_id: Optional[int],
        config: MCTSConfig,
    ) -> None:
        self._config = config
        if self.ice_board is None and (config.use_ice_pruning or config.use_vcs) and self.ice_engine is not None:
            self.ice_board = Board(np.zeros(TOTAL_CELLS, dtype=np.uint8), self.ice_engine, config)
        
        self.playout_phase.set_root_store_from_board(board_state)
        if self._gamma_calculator is None:
            self._gamma_calculator = GammaPriorCalculator(self.playout_phase.root_store)
        else:
            self._gamma_calculator.set_store(self.playout_phase.root_store)
        
        initial_actions: NDArray[np.int32] = empty_tracker.possible_actions().astype(np.int32, copy=True)
        ice_pruned_mask: NDArray[np.bool_] = np.ones(TOTAL_CELLS, dtype=bool)
        
        if (config.use_ice_pruning or config.use_vcs) and self.ice_engine is not None:
            ice_pruned_mask = self._compute_consider_set_mask(board_state, starting_player, last_move_id, config)
            num_valid = int(np.sum(ice_pruned_mask[initial_actions]))
            num_pruned = int(initial_actions.size) - num_valid
            pruned_moves = initial_actions[~ice_pruned_mask[initial_actions]]
            print(f"CONSIDER_SET_ROOT: before={int(initial_actions.size)}, after={num_valid}, pruned={num_pruned}, moves={pruned_moves.tolist()}")
        
        for it in range(num_iterations):
            logger: Optional[IterationLogger] = None
            if config.detailed_iteration_logging:
                logger = IterationLogger(config.log_dir, it)
            
            self._run_single_iteration(
                flat_tree, root_index, starting_player, board_state,
                empty_tracker, clusters, zobrist, last_move_id, config,
                logger, ice_pruned_mask
            )

    def _compute_consider_set_mask(
        self,
        board_state: NDArray[np.int_],
        color_to_move: int,
        last_move: Optional[int],
        config: MCTSConfig,
    ) -> NDArray[np.bool_]:
        if self.ice_board is None:
            return np.ones(TOTAL_CELLS, dtype=bool)
        
        self._sync_ice_board(board_state)
        
        last_move_arg: Optional[int] = last_move if last_move is not None else -1
        if last_move_arg == -1:
            last_move_arg = None
        
        reverser_index: Optional[int] = self.ice_board.compute_all(
            color_to_move,
            last_move=last_move_arg,
            add_fillin=False,
            only_around_last_move=False,
        )
        
        consider_mask: NDArray[np.bool_] = np.zeros(self.ice_board.n_cells, dtype=bool)
        
        if config.use_vcs and self.ice_board.use_vcs():
            mustplay_mask: NDArray[np.bool_] = self.ice_board.get_mustplay(color_to_move)
            num_mustplay: int = int(np.sum(mustplay_mask))
            if num_mustplay > 0:
                consider_mask = mustplay_mask.copy()
            else:
                consider_mask = np.ones(self.ice_board.n_cells, dtype=bool)
        else:
            consider_mask = np.ones(self.ice_board.n_cells, dtype=bool)
        
        empty_mask: NDArray[np.bool_] = self.ice_board.board_state == 0
        consider_mask = consider_mask & empty_mask
        
        inferior_cells_obj = self.ice_board.get_inferior_cells()
        inferior_mask: NDArray[np.bool_] = np.zeros(self.ice_board.n_cells, dtype=bool)
        inferior_mask |= inferior_cells_obj.vulnerable()
        inferior_mask |= inferior_cells_obj.s_reversible()
        inferior_mask |= inferior_cells_obj.inferior()
        
        if reverser_index is not None:
            inferior_mask[reverser_index] = False
        
        consider_mask = consider_mask & ~inferior_mask
        
        return consider_mask
    
    def _compute_ice_pruning_mask(
        self,
        board_state: NDArray[np.int_],
        color_to_move: int,
        last_move: Optional[int],
    ) -> NDArray[np.bool_]:
        if self.ice_board is None:
            return np.ones(TOTAL_CELLS, dtype=bool)
        
        self._sync_ice_board(board_state)
        
        reverser_index: Optional[int] = self.ice_board.compute_inferior_cells(color_to_move, last_move=last_move)
        
        inferior_cells_obj = self.ice_board.get_inferior_cells()
        inferior_mask: NDArray[np.bool_] = np.zeros(self.ice_board.n_cells, dtype=bool)
        inferior_mask |= inferior_cells_obj.vulnerable()
        inferior_mask |= inferior_cells_obj.s_reversible()
        inferior_mask |= inferior_cells_obj.inferior()
        
        if reverser_index is not None:
            inferior_mask[reverser_index] = False
        
        valid_mask: NDArray[np.bool_] = ~inferior_mask
        return valid_mask

    def _sync_ice_board(self, board_state: NDArray[np.int_]) -> None:
        if self.ice_board is None:
            return
        
        board_uint8 = board_state.astype(np.uint8)
        diff_mask: NDArray[np.bool_] = self.ice_board.board_state != board_uint8
        
        if np.any(diff_mask):
            changed_indices: NDArray[np.intp] = np.flatnonzero(diff_mask)
            self.ice_board.board_state[changed_indices] = board_uint8[changed_indices]
            self.ice_board.groups = GroupBuilder.build(self.ice_board.board_state)
            self.ice_board.pastate = PatternState(self.ice_board.board_state)
            self.ice_board.clear_inferior_cells()

    def _run_single_iteration(
        self,
        flat_tree: FlatTree,
        root_index: int,
        starting_player: int,
        board_state: NDArray[np.int_],
        empty_tracker: IncrementalEmptyCellsTracker,
        clusters: Clusters,
        zobrist: Zobrist,
        last_move_id: Optional[int],
        config: MCTSConfig,
        logger: Optional[IterationLogger],
        ice_pruned_mask: NDArray[np.bool_],
    ) -> None:
        path: List[int] = [root_index]

        board: NDArray[np.int_] = board_state.copy()
        available_moves: NDArray[np.int32] = empty_tracker.possible_actions().astype(np.int32, copy=True)
        active_count: int = int(available_moves.size)
        move_to_index: NDArray[np.int32] = np.full(TOTAL_CELLS, -1, dtype=np.int32)
        if active_count > 0:
            move_to_index[available_moves[:active_count]] = np.arange(active_count, dtype=np.int32)
        
        temp_clusters = clusters.copy()
        current_node_idx = root_index
        current_player = starting_player
        last_move_local: Optional[int] = last_move_id
        
        rave_actions: List[int] = []
        actions_taken_ids = np.empty(TOTAL_CELLS, dtype=np.int32)
        actions_len = 0

        if logger is not None:
            logger.write(f"=== ITERATION ===")
            logger.write("=== BOARD_START ===")
            logger.write_block("", board_to_hex_ascii(board, None, False))
            logger.write("=== END BOARD_START ===")
        
        while True:
            if active_count == 0:
                break
            
            possible_actions = available_moves[:active_count]
            final_actions_for_selection = possible_actions
            
            has_knowledge = bool(flat_tree.knowledge_computed[current_node_idx])
            
            if current_node_idx == root_index:
                ice_filtered = final_actions_for_selection[ice_pruned_mask[final_actions_for_selection]]
                if ice_filtered.size > 0:
                    final_actions_for_selection = ice_filtered
            elif has_knowledge:
                knowledge_mask = flat_tree.knowledge_mask[current_node_idx, final_actions_for_selection]
                knowledge_filtered = final_actions_for_selection[knowledge_mask]
                if knowledge_filtered.size > 0:
                    final_actions_for_selection = knowledge_filtered
            elif flat_tree.parent_visit_sum[current_node_idx] > config.knowledge_threshold:
                before_knowledge = int(final_actions_for_selection.size)
                last_move_for_node: Optional[int] = -1 if last_move_local is None else last_move_local
                knowledge_mask = self._compute_consider_set_mask(board, current_player, last_move_for_node, config)
                flat_tree.knowledge_mask[current_node_idx, :] = knowledge_mask
                flat_tree.knowledge_computed[current_node_idx] = True
                
                flat_tree.priors[current_node_idx, ~knowledge_mask] = 0.0
                
                knowledge_filtered = final_actions_for_selection[knowledge_mask[final_actions_for_selection]]
                if knowledge_filtered.size > 0:
                    final_actions_for_selection = knowledge_filtered
                pruned_by_knowledge = before_knowledge - int(final_actions_for_selection.size)
                print(f"NODE_PRUNED_AT_THRESHOLD: node={current_node_idx}, parent_visits={flat_tree.parent_visit_sum[current_node_idx]}, actions_pruned={pruned_by_knowledge}")
            
            if final_actions_for_selection.size == 0:
                if logger is not None:
                    logger.write(f"=== PROVEN_LOSS (node={current_node_idx}) ===")
                rollout_value = PROVEN_LOSS
                self._backpropagate(flat_tree, path, actions_taken_ids, actions_len, rollout_value, rave_actions, logger)
                return
            
            if not bool(flat_tree.priors_initialized[current_node_idx]):
                final_actions_for_selection = self._initialize_priors(
                    flat_tree, current_node_idx, current_player, final_actions_for_selection,
                    last_move_local, config, logger
                )
            
            parent_sum = flat_tree.get_parent_visit_sum(current_node_idx)
            rave_skip = self.selection_phase.update_rave_skip_counter()
            action_id = self.selection_phase.select_best_action(
                flat_tree.visit_count[current_node_idx],
                flat_tree.q_value[current_node_idx],
                flat_tree.priors[current_node_idx],
                parent_sum,
                final_actions_for_selection,
                flat_tree.rave_q[current_node_idx] if flat_tree.rave_q is not None else None,
                flat_tree.rave_visit[current_node_idx] if flat_tree.rave_visit is not None else None,
                rave_skip=rave_skip
            )
            
            if logger is not None:
                logger.write(f"=== SELECT (node={current_node_idx}, parentN={parent_sum}, chose={int(action_id)}, from={int(final_actions_for_selection.size)}) ===")
            
            actions_taken_ids[actions_len] = action_id
            actions_len += 1

            if int(flat_tree.children[current_node_idx, action_id]) == -1:
                self._expand_node(
                    flat_tree, board, available_moves, move_to_index,
                    temp_clusters, current_node_idx, action_id, current_player,
                    zobrist, logger
                )
                current_node_idx = int(flat_tree.children[current_node_idx, action_id])
                active_count -= 1
                current_player = 3 - current_player
                last_move_local = action_id
                break
            else:
                self._step_down_node(
                    flat_tree, board, available_moves, move_to_index,
                    temp_clusters, current_node_idx, action_id, current_player,
                    logger
                )
                path.append(int(flat_tree.children[current_node_idx, action_id]))
                current_node_idx = int(flat_tree.children[current_node_idx, action_id])
                active_count -= 1
                current_player = 3 - current_player
                last_move_local = action_id

        rollout_value, rollout_rave_actions = self.playout_phase.execute_rollout(
            current_player,
            board,
            available_moves[:active_count],
            temp_clusters,
            logger=logger,
        )
        rave_actions.extend(rollout_rave_actions)

        self._backpropagate(
            flat_tree, path, actions_taken_ids, actions_len,
            rollout_value, rave_actions, logger
        )

    def _initialize_priors(
        self,
        flat_tree: FlatTree,
        node_idx: int,
        current_player: int,
        possible_actions: NDArray[np.int32],
        last_move_local: Optional[int],
        config: MCTSConfig,
        logger: Optional[IterationLogger],
    ) -> NDArray[np.int32]:
        if config.use_pattern_priors and self._gamma_calculator is not None:
            pri = self._gamma_calculator.compute_priors(
                int(current_player), possible_actions, last_move_local,
                float(config.gamma_pruning_threshold)
            )
            if pri.size != possible_actions.size:
                pri = np.zeros(possible_actions.size, dtype=np.float32)
            keep_mask = pri > 0.0
            if not bool(np.any(keep_mask)):
                pri = np.full(possible_actions.size, 1.0 / float(max(1, possible_actions.size)), dtype=np.float32)
                keep_mask = np.ones(possible_actions.size, dtype=bool)
            flat_tree.priors[node_idx, possible_actions] = pri
            pruned_by_gamma = int(np.sum(~keep_mask))
            if logger is not None and pruned_by_gamma > 0:
                pruned_moves = possible_actions[~keep_mask]
                logger.write(f"=== GAMMA_PRUNING (node={node_idx}, before={int(possible_actions.size)}, after={int(np.sum(keep_mask))}, pruned={pruned_by_gamma}, moves={pruned_moves.tolist()}) ===")
            possible_actions = possible_actions[keep_mask]
            flat_tree.priors_initialized[node_idx] = True
            
            if logger is not None:
                pri_log = flat_tree.priors[node_idx, possible_actions]
                logger.write(f"=== PRIORS (node={node_idx}, size={int(possible_actions.size)}) ===")
                logger.write(str(pri_log.tolist()))
                logger.write("=== END PRIORS ===")
        
        return possible_actions

    def _expand_node(
        self,
        flat_tree: FlatTree,
        board: NDArray[np.int_],
        available_moves: NDArray[np.int32],
        move_to_index: NDArray[np.int32],
        temp_clusters: Clusters,
        current_node_idx: int,
        action_id: int,
        current_player: int,
        zobrist: Zobrist,
        logger: Optional[IterationLogger],
    ) -> None:
        next_hash, next_player_piece = zobrist.apply_move_hash(
            int(flat_tree.z_hash[current_node_idx]),
            int(flat_tree.player_piece[current_node_idx]),
            int(action_id)
        )
        flat_tree.ensure_child(current_node_idx, action_id, int(next_hash), int(next_player_piece))
        
        self._apply_move(board, available_moves, move_to_index, action_id, current_player, temp_clusters)
        
        if logger is not None:
            logger.write(f"=== BOARD_AFTER_EXPANSION (node={current_node_idx}, action={action_id}, player={3 - current_player}) ===")
            logger.write_block("", board_to_hex_ascii(board, action_id, False))
            logger.write("=== END BOARD_AFTER_EXPANSION ===")

    def _step_down_node(
        self,
        flat_tree: FlatTree,
        board: NDArray[np.int_],
        available_moves: NDArray[np.int32],
        move_to_index: NDArray[np.int32],
        temp_clusters: Clusters,
        current_node_idx: int,
        action_id: int,
        current_player: int,
        logger: Optional[IterationLogger],
    ) -> None:
        self._apply_move(board, available_moves, move_to_index, action_id, current_player, temp_clusters)
        
        if logger is not None:
            logger.write(f"=== BOARD_STEP_DOWN (node={current_node_idx}, action={action_id}, player={3 - current_player}) ===")
            logger.write_block("", board_to_hex_ascii(board, action_id, False))
            logger.write("=== END BOARD_STEP_DOWN ===")

    def _apply_move(
        self,
        board: NDArray[np.int_],
        available_moves: NDArray[np.int32],
        move_to_index: NDArray[np.int32],
        action_id: int,
        current_player: int,
        temp_clusters: Clusters,
    ) -> None:
        temp_clusters.add_piece_and_update_groups(action_id, current_player)
        board[action_id] = current_player
        
        rem_idx = int(move_to_index[action_id])
        last_idx = len(available_moves) - 1
        last_val = int(available_moves[last_idx])
        available_moves[rem_idx] = last_val
        move_to_index[last_val] = rem_idx
        move_to_index[action_id] = -1

    def _backpropagate(
        self,
        flat_tree: FlatTree,
        path: List[int],
        actions_taken_ids: NDArray[np.int32],
        actions_len: int,
        rollout_value: float,
        rave_actions: List[int],
        logger: Optional[IterationLogger],
    ) -> None:
        value: float = float(rollout_value)
        path_len = len(path)
        rave_aids: Optional[NDArray[np.int32]] = None
        
        if flat_tree.use_rave and flat_tree.rave_visit is not None and flat_tree.rave_q is not None and len(rave_actions) > 0:
            rave_aids = np.asarray(rave_actions, dtype=np.int32)
        
        for i in reversed(range(path_len)):
            node_idx = path[i]
            if i < actions_len:
                action = int(actions_taken_ids[i])
                n = int(flat_tree.visit_count[node_idx, action]) + 1
                flat_tree.q_value[node_idx, action] = (flat_tree.q_value[node_idx, action] * (n - 1) + value) / n
                flat_tree.visit_count[node_idx, action] = n
                flat_tree.parent_visit_sum[node_idx] += 1
                
                if rave_aids is not None and flat_tree.rave_visit is not None and flat_tree.rave_q is not None:
                    old_counts: NDArray[np.int32] = flat_tree.rave_visit[node_idx, rave_aids]
                    old_qs: NDArray[np.float32] = flat_tree.rave_q[node_idx, rave_aids]
                    np.multiply(old_qs, old_counts, out=old_qs)
                    np.add(old_qs, value, out=old_qs)
                    np.divide(old_qs, old_counts + 1, out=old_qs)
                    flat_tree.rave_visit[node_idx, rave_aids] = old_counts + 1
                    flat_tree.rave_q[node_idx, rave_aids] = old_qs
            
            value = -value
        
        if logger is not None:
            logger.write(f"ROLLOUT_VALUE {rollout_value:.4f}")
            logger.write("=== BOARD_END ===")
            logger.flush()