import numpy as np
from typing import List, Optional, Set
from collections import deque
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP
from src_2206596_2122457.src.group_system.groups import Group, Groups, EMPTY, BLUE, RED, BOARD_SIZE, TOTAL_CELLS


def _build_adjacency_matrix() -> np.ndarray:
    adjacency: np.ndarray = np.zeros((TOTAL_CELLS, TOTAL_CELLS), dtype=bool)
    
    for cell_a in range(TOTAL_CELLS):
        neighbors_raw: np.ndarray = NEIGHBORS_LOOKUP[cell_a]
        valid_neighbors: np.ndarray = neighbors_raw[neighbors_raw != -1]
        adjacency[cell_a, valid_neighbors] = True
    
    return adjacency


ADJACENCY_MATRIX: np.ndarray = _build_adjacency_matrix()


def are_cells_adjacent_fast(cell_a: int, cell_b: int) -> bool:
    return bool(ADJACENCY_MATRIX[cell_a, cell_b])


def are_cells_adjacent_vectorized(cells_a: np.ndarray, cells_b: np.ndarray) -> np.ndarray:
    return ADJACENCY_MATRIX[cells_a, cells_b]


class DeadRegion:

    @staticmethod
    def compute_edge_isolated_regions(
        board_state: np.ndarray,
        player_color: int,
        stop_set: np.ndarray,
        allow_flow_from_player_edge: bool,
        allow_flow_from_opposite_edge: bool,
    ) -> np.ndarray:
        flow_set: np.ndarray = (board_state == EMPTY) | (board_state == player_color)
        reachable_cells: np.ndarray = np.zeros_like(board_state, dtype=bool)
        
        if allow_flow_from_player_edge:
            player_edge_indices: np.ndarray = compute_edge_indices_for_player(
                player_color
            )
            perform_bfs_from_edges(
                player_edge_indices, flow_set, stop_set, reachable_cells
            )
        
        if allow_flow_from_opposite_edge:
            opposite_player_color: int = RED if player_color == BLUE else BLUE
            opposite_edge_indices: np.ndarray = compute_edge_indices_for_player(
                opposite_player_color
            )
            perform_bfs_from_edges(
                opposite_edge_indices, flow_set, stop_set, reachable_cells
            )
        
        empty_cells_mask: np.ndarray = board_state == EMPTY
        dead_region: np.ndarray = empty_cells_mask & ~reachable_cells
        
        return dead_region

    @staticmethod
    def compute_group_dead_regions(groups: Groups) -> np.ndarray:
        dead_region_mask: np.ndarray = np.zeros(TOTAL_CELLS, dtype=bool)
        
        if groups.is_game_over():
            return groups.board_state == EMPTY
        
        for group_instance in groups.groups:
            if group_instance.size() <= 1:
                continue
            
            barrier_stop_set: np.ndarray = group_instance.neighbors.copy()
            
            should_allow_player_edge_flow: bool = is_captain_in_upper_half(
                group_instance.captain
            )
            should_allow_opposite_edge_flow: bool = (
                not should_allow_player_edge_flow
            )
            
            group_dead_region: np.ndarray = DeadRegion.compute_edge_isolated_regions(
                groups.board_state,
                group_instance.color,
                barrier_stop_set,
                should_allow_player_edge_flow,
                should_allow_opposite_edge_flow,
            )
            
            dead_region_mask |= group_dead_region
        
        return dead_region_mask

    @staticmethod
    def compute_dead_regions_single_group(
        board_state: np.ndarray, groups: Groups
    ) -> np.ndarray:
        dead_region_mask: np.ndarray = np.zeros(TOTAL_CELLS, dtype=bool)
        empty_cell_indices: np.ndarray = np.where(board_state == EMPTY)[0]
        
        for index_x, cell_x in enumerate(empty_cell_indices):
            for index_y in range(index_x + 1, len(empty_cell_indices)):
                cell_y: int = empty_cell_indices[index_y]
                
                if are_cells_adjacent_fast(cell_x, cell_y):
                    continue
                
                group_containing_x: Group = groups.get_group(cell_x)
                group_containing_y: Group = groups.get_group(cell_y)
                neighbors_of_x: np.ndarray = group_containing_x.neighbors
                neighbors_of_y: np.ndarray = group_containing_y.neighbors
                
                common_neighbor_groups: np.ndarray = neighbors_of_x & neighbors_of_y
                if not np.any(common_neighbor_groups):
                    continue
                
                common_group_indices_set: Set[int] = set()
                for neighbor_cell_index in np.where(common_neighbor_groups)[0]:
                    neighbor_group_index: int = int(groups.group_index[neighbor_cell_index])
                    common_group_indices_set.add(neighbor_group_index)
                
                for cell_z in empty_cell_indices:
                    if cell_z == cell_x or cell_z == cell_y:
                        continue
                    
                    if not (are_cells_adjacent_fast(cell_x, cell_z) and 
                            are_cells_adjacent_fast(cell_y, cell_z)):
                        continue
                    
                    group_containing_z: Group = groups.get_group(cell_z)
                    z_neighbor_group_indices_set: Set[int] = set()
                    for neighbor_cell_index in np.where(group_containing_z.neighbors)[0]:
                        neighbor_group_index: int = int(groups.group_index[neighbor_cell_index])
                        z_neighbor_group_indices_set.add(neighbor_group_index)
                    
                    remaining_barrier_groups: Set[int] = (
                        common_group_indices_set - z_neighbor_group_indices_set
                    )
                    
                    if remaining_barrier_groups:
                        barrier_cell_set: np.ndarray = np.zeros(TOTAL_CELLS, dtype=bool)
                        barrier_cell_set[cell_x] = True
                        barrier_cell_set[cell_y] = True
                        barrier_cell_set[cell_z] = True
                        
                        for player_color in [BLUE, RED]:
                            barrier_dead_region: np.ndarray = (
                                DeadRegion.compute_edge_isolated_regions(
                                    board_state,
                                    player_color,
                                    barrier_cell_set,
                                    allow_flow_from_player_edge=True,
                                    allow_flow_from_opposite_edge=False,
                                )
                            )
                            dead_region_mask |= barrier_dead_region
        
        return dead_region_mask

    @staticmethod
    def compute_dead_regions_two_group_interaction(
        board_state: np.ndarray, groups: Groups
    ) -> np.ndarray:
        dead_region_mask: np.ndarray = np.zeros(TOTAL_CELLS, dtype=bool)
        
        for player_color in [BLUE, RED]:
            groups_of_same_color: List[Group] = [
                group_instance for group_instance in groups.groups if group_instance.color == player_color
            ]
            
            for group_index_i, group_one in enumerate(groups_of_same_color):
                if is_group_on_edge(group_one.captain):
                    continue
                
                for group_two in groups_of_same_color[group_index_i + 1 :]:
                    if is_group_on_edge(group_two.captain):
                        continue
                    
                    group_one_empty_neighbors: np.ndarray = (
                        group_one.neighbors & (board_state == EMPTY)
                    )
                    group_two_empty_neighbors: np.ndarray = (
                        group_two.neighbors & (board_state == EMPTY)
                    )
                    
                    common_empty_neighbors: np.ndarray = (
                        group_one_empty_neighbors & group_two_empty_neighbors
                    )
                    group_one_exclusive_empty: np.ndarray = (
                        group_one_empty_neighbors & ~group_two_empty_neighbors
                    )
                    group_two_exclusive_empty: np.ndarray = (
                        group_two_empty_neighbors & ~group_one_empty_neighbors
                    )
                    
                    if not np.any(common_empty_neighbors):
                        continue
                    
                    group_one_exclusive_indices: np.ndarray = np.where(group_one_exclusive_empty)[0]
                    group_two_exclusive_indices: np.ndarray = np.where(group_two_exclusive_empty)[0]
                    
                    for exclusive_cell_from_group_one in group_one_exclusive_indices:
                        for exclusive_cell_from_group_two in group_two_exclusive_indices:
                            if not are_cells_adjacent_fast(exclusive_cell_from_group_one, exclusive_cell_from_group_two):
                                continue
                            
                            barrier_cell_set: np.ndarray = common_empty_neighbors.copy()
                            barrier_cell_set[exclusive_cell_from_group_one] = True
                            barrier_cell_set[exclusive_cell_from_group_two] = True
                            
                            barrier_dead_region: np.ndarray = (
                                DeadRegion.compute_edge_isolated_regions(
                                    board_state,
                                    player_color,
                                    barrier_cell_set,
                                    allow_flow_from_player_edge=True,
                                    allow_flow_from_opposite_edge=False,
                                )
                            )
                            dead_region_mask |= barrier_dead_region
        
        return dead_region_mask

    @staticmethod
    def compute_dead_regions_three_group_interaction(
        board_state: np.ndarray, groups: Groups
    ) -> np.ndarray:
        dead_region_mask: np.ndarray = np.zeros(TOTAL_CELLS, dtype=bool)
        
        for player_color in [BLUE, RED]:
            groups_of_same_color: List[Group] = [
                group_instance for group_instance in groups.groups if group_instance.color == player_color
            ]
            
            for group_index_i, group_one in enumerate(groups_of_same_color):
                if is_group_on_edge(group_one.captain):
                    continue
                
                for group_index_j, group_two in enumerate(groups_of_same_color[group_index_i + 1 :], start=group_index_i + 1):
                    if is_group_on_edge(group_two.captain):
                        continue
                    
                    group_one_and_two_common_empty: np.ndarray = (
                        group_one.neighbors & group_two.neighbors & (board_state == EMPTY)
                    )
                    if not np.any(group_one_and_two_common_empty):
                        continue
                    
                    for group_three in groups_of_same_color[group_index_j + 1 :]:
                        if is_group_on_edge(group_three.captain):
                            continue
                        
                        group_one_and_three_common_empty: np.ndarray = (
                            group_one.neighbors & group_three.neighbors & (board_state == EMPTY)
                        )
                        group_two_and_three_common_empty: np.ndarray = (
                            group_two.neighbors & group_three.neighbors & (board_state == EMPTY)
                        )
                        
                        if not np.any(group_one_and_three_common_empty) or not np.any(group_two_and_three_common_empty):
                            continue
                        
                        barrier_cell_set: np.ndarray = (
                            group_one_and_two_common_empty | group_one_and_three_common_empty | group_two_and_three_common_empty
                        )
                        
                        barrier_dead_region: np.ndarray = (
                            DeadRegion.compute_edge_isolated_regions(
                                board_state,
                                player_color,
                                barrier_cell_set,
                                allow_flow_from_player_edge=True,
                                allow_flow_from_opposite_edge=False,
                            )
                        )
                        dead_region_mask |= barrier_dead_region
        
        return dead_region_mask

    @staticmethod
    def compute_dead_regions_multi_group_interaction(
        groups: Groups,
    ) -> np.ndarray:
        if groups.is_game_over():
            return groups.board_state == EMPTY
        
        single_group_dead_region: np.ndarray = (
            DeadRegion.compute_dead_regions_single_group(groups.board_state, groups)
        )
        two_group_dead_region: np.ndarray = (
            DeadRegion.compute_dead_regions_two_group_interaction(
                groups.board_state, groups
            )
        )
        three_group_dead_region: np.ndarray = (
            DeadRegion.compute_dead_regions_three_group_interaction(
                groups.board_state, groups
            )
        )
        
        aggregated_dead_region: np.ndarray = single_group_dead_region | two_group_dead_region | three_group_dead_region
        
        return aggregated_dead_region

    @staticmethod
    def verify_complete_adjacency(
        cell_indices: np.ndarray, exclude: Optional[int] = None
    ) -> bool:
        valid_cell_indices: np.ndarray = cell_indices[cell_indices != -1]
        
        if exclude is not None:
            valid_cell_indices = valid_cell_indices[valid_cell_indices != exclude]
        
        for index_i in range(len(valid_cell_indices)):
            for index_j in range(index_i + 1, len(valid_cell_indices)):
                if not are_cells_adjacent_fast(
                    int(valid_cell_indices[index_i]), int(valid_cell_indices[index_j])
                ):
                    return False
        
        return True


def compute_edge_indices_for_player(player_color: int) -> np.ndarray:
    edge_cell_indices: List[int] = []
    
    if player_color == BLUE:
        for column_index in range(BOARD_SIZE):
            edge_cell_indices.append(column_index)
            edge_cell_indices.append((BOARD_SIZE - 1) * BOARD_SIZE + column_index)
    else:
        for row_index in range(BOARD_SIZE):
            edge_cell_indices.append(row_index * BOARD_SIZE)
            edge_cell_indices.append(row_index * BOARD_SIZE + BOARD_SIZE - 1)
    
    return np.array(edge_cell_indices, dtype=np.int32)


def perform_bfs_from_edges(
    starting_edge_indices: np.ndarray,
    traversable_cells_mask: np.ndarray,
    blocked_cells_mask: np.ndarray,
    reachable_cells_output: np.ndarray,
) -> None:
    cell_queue: deque[int] = deque()
    
    for edge_cell_index in starting_edge_indices:
        if traversable_cells_mask[edge_cell_index] and not blocked_cells_mask[edge_cell_index]:
            cell_queue.append(int(edge_cell_index))
            reachable_cells_output[int(edge_cell_index)] = True
    
    while cell_queue:
        current_cell_index: int = cell_queue.popleft()
        neighbor_cells: np.ndarray = NEIGHBORS_LOOKUP[current_cell_index]
        
        for neighbor_cell_index in neighbor_cells:
            if neighbor_cell_index == -1:
                continue
            
            neighbor_cell_index_int: int = int(neighbor_cell_index)
            
            if reachable_cells_output[neighbor_cell_index_int]:
                continue
            
            if not traversable_cells_mask[neighbor_cell_index_int]:
                continue
            
            if blocked_cells_mask[neighbor_cell_index_int]:
                continue
            
            reachable_cells_output[neighbor_cell_index_int] = True
            cell_queue.append(neighbor_cell_index_int)


def is_captain_in_upper_half(captain_cell_index: int) -> bool:
    captain_row_index: int = captain_cell_index // BOARD_SIZE
    return captain_row_index < BOARD_SIZE // 2


def is_group_on_edge(captain_cell_index: int) -> bool:
    captain_row_index: int = captain_cell_index // BOARD_SIZE
    captain_column_index: int = captain_cell_index % BOARD_SIZE
    
    return (
        captain_row_index == 0
        or captain_row_index == BOARD_SIZE - 1
        or captain_column_index == 0
        or captain_column_index == BOARD_SIZE - 1
    )