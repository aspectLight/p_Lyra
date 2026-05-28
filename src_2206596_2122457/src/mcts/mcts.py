from typing import Tuple, Optional, Dict
import numpy as np

from game_state_hex import GameStateHex
from src_2206596_2122457.src.config import MCTSConfig
from src_2206596_2122457.src.constants import BOARD_SIZE, TOTAL_CELLS, PIECE_B, PIECE_R
from src_2206596_2122457.src.util.board_utils import action_to_id
from src_2206596_2122457.src.game_trackers.game_state_manager import GameStateManager
from src_2206596_2122457.src.game_trackers.player_identity import PlayerIdentity
from src_2206596_2122457.src.mcts.search_executor import SearchExecutor
from src_2206596_2122457.src.mcts.selection_phase import SelectionPhase
from src_2206596_2122457.src.mcts.playout_phase import PlayoutPhase
from src_2206596_2122457.src.util.tree_manager import TreeManager
from src_2206596_2122457.src.cluster_system.clusters import Clusters
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP
from src_2206596_2122457.src.ice_system.ice_engine.ice_engine import ICEngine
from seahorse.game.game_layout.board import Piece


def compute_edge_flags() -> np.ndarray:
    flags = np.zeros((TOTAL_CELLS, 4), dtype=np.bool_)
    for idx in range(TOTAL_CELLS):
        q = idx // BOARD_SIZE
        r = idx % BOARD_SIZE
        flags[idx, 0] = (q == 0)
        flags[idx, 1] = (q == BOARD_SIZE - 1)
        flags[idx, 2] = (r == 0)
        flags[idx, 3] = (r == BOARD_SIZE - 1)
    return flags


def extract_board_array(env: Dict[Tuple[int, int], Piece]) -> np.ndarray:
    board = np.zeros(TOTAL_CELLS, dtype=np.int8)
    for (i, j), piece in env.items():
        pid: str = piece.get_type()
        board[i * BOARD_SIZE + j] = PIECE_R if pid == "R" else PIECE_B
    return board


def id_to_action(action_id: int) -> Tuple[int, int]:
    return action_id // BOARD_SIZE, action_id % BOARD_SIZE


class MCTS:
    def __init__(self, config: MCTSConfig):
        self.config = config

        self.selection_phase = SelectionPhase(config)
        self.playout_phase = PlayoutPhase(config)
        
        ice_engine: Optional[ICEngine] = None
        if config.use_ice_pruning:
            ice_engine = self._initialize_ice_engine(config)
        elif config.use_vcs:
            engine = ICEngine()
            engine.load_patterns()
            ice_engine = engine
        
        self.search_executor = SearchExecutor(
            self.selection_phase,
            self.playout_phase,
            ice_engine=ice_engine,
        )

        self.state_manager = GameStateManager()
        self.tree_manager = TreeManager(config)

    def _initialize_ice_engine(self, config: MCTSConfig) -> ICEngine:
        engine = ICEngine()
        engine.set_find_presimplicial_pairs(config.ice_find_presimplicial_pairs)
        engine.set_find_all_pattern_killers(config.ice_find_all_pattern_killers)
        engine.set_find_all_pattern_superiors(config.ice_find_all_pattern_superiors)
        engine.set_find_three_sided_dead_regions(config.ice_find_three_sided_dead_regions)
        engine.set_iterative_dead_regions(config.ice_iterative_dead_regions)
        engine.set_use_capture(config.ice_use_capture)
        engine.set_find_reversible(config.ice_find_reversible)
        engine.set_use_s_reversible_as_reversible(config.ice_use_s_reversible_as_reversible)
        engine.load_patterns()
        return engine

    def get_action(self, state: GameStateHex, player_identity: PlayerIdentity) -> Tuple[int, int]:
        our_piece_type = player_identity.get_our_piece_type()
        our_piece_letter = player_identity.get_our_piece_letter()

        env: Dict[Tuple[int, int], Piece] = state.get_rep().get_env() # type: ignore
        self.state_manager.sync_with_opponent_move(env, player_identity)

        board_array = extract_board_array(env)
        edge_flags = compute_edge_flags()
        clusters = Clusters(BOARD_SIZE, NEIGHBORS_LOOKUP, edge_flags)
        clusters.build_from_board_state(board_array)

        base_hash = self.tree_manager.zobrist.compute_hash_from_array(board_array)
        current_root_hash = self.tree_manager.zobrist.xor_player(base_hash, our_piece_type)

        opponent_move_id: Optional[int] = None
        if self.tree_manager.get_root() is None:
            self.tree_manager.initialize_root(current_root_hash, our_piece_type)
        else:
            opponent_move = self.state_manager.last_move_tracker.compute_opponent_move(
                env, our_piece_letter
            )
            if opponent_move is not None:
                opponent_move_id = action_to_id(opponent_move)
            self.tree_manager.advance_on_opponent_move(opponent_move_id, current_root_hash)

        root_index = self.tree_manager.get_root()
        assert root_index is not None
        flat_tree = self.tree_manager.get_flat_tree()

        self.search_executor.run_search_iterations(
            flat_tree=flat_tree,
            root_index=root_index,
            num_iterations=self.config.num_simulations,
            starting_player=our_piece_type,
            board_state=board_array,
            empty_tracker=self.state_manager.empty_cells_tracker,
            clusters=clusters,
            zobrist=self.tree_manager.zobrist,
            last_move_id=opponent_move_id,
            config=self.config,
        )

        empty_cells = self.state_manager.empty_cells_tracker.possible_actions()
        visits = flat_tree.visit_count[root_index, empty_cells]
        max_visits = visits.max()
        best_indices = np.flatnonzero(visits == max_visits)
        chosen_idx = np.random.choice(best_indices)
        best_action_id = int(empty_cells[chosen_idx])
        best_position = id_to_action(best_action_id)

        self.state_manager.empty_cells_tracker.remove(best_action_id)
        self.tree_manager.advance_on_our_move(best_action_id, current_root_hash)
        self.state_manager.last_move_tracker.set_our_board(env)

        return best_position