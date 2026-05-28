import numpy as np
from typing import Dict, List, Set
from src_2206596_2122457.src.precomputed.neighbors_lookup import NEIGHBORS_LOOKUP

EMPTY = 0
BLUE = 1
RED = 2

BOARD_SIZE = 14
TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE


def get_neighbors(point: int) -> np.ndarray:
    neighbor_data: np.ndarray = NEIGHBORS_LOOKUP[point]
    mask: np.ndarray = neighbor_data != -1
    return neighbor_data[mask]


class Group:
    def __init__(
        self,
        color: int,
        captain: int,
        members: np.ndarray,
        neighbors: np.ndarray,
        groups_ref: 'Groups',
    ) -> None:
        self.color: int = color
        self.captain: int = captain
        self.members: np.ndarray = members.copy()
        self.neighbors: np.ndarray = neighbors.copy()
        self.neighbor_indices: List[int] = []
        self.colorset_neighbors: Dict[int, np.ndarray] = {}
        self.groups_ref: Groups = groups_ref
        self._size: int = int(np.sum(self.members))

    def size(self) -> int:
        return self._size

    def is_member(self, point_index: int) -> bool:
        return bool(self.members[point_index])

    def neighbors_for_colorset(self, colorset: Set[int]) -> np.ndarray:
        key: int = frozenset(colorset).__hash__()
        if key not in self.colorset_neighbors:
            self._compute_colorset_neighbors_helper(colorset)
        return self.colorset_neighbors[key]

    def _compute_colorset_neighbors_helper(self, colorset: Set[int]) -> None:
        result: np.ndarray = np.zeros(len(self.groups_ref.groups), dtype=bool)
        for neighbor_idx in self.neighbor_indices:
            neighbor_group: Group = self.groups_ref.groups[neighbor_idx]
            if neighbor_group.color in colorset:
                result[neighbor_idx] = True
        key: int = frozenset(colorset).__hash__()
        self.colorset_neighbors[key] = result

    def compute_colorset_neighbors(self) -> None:
        all_colors: Set[int] = {BLUE, RED}
        for color in all_colors:
            self._compute_colorset_neighbors_helper({color})


class Groups:
    def __init__(self, board_state: np.ndarray) -> None:
        self.board_state: np.ndarray = board_state
        self.groups: List[Group] = []
        self.group_index: np.ndarray = np.zeros_like(board_state, dtype=np.int32)
        self._build_groups()

    def _build_groups(self) -> None:
        visited: np.ndarray = np.zeros_like(self.board_state, dtype=bool)
        group_count: int = 0

        for point in range(len(self.board_state)):
            if visited[point]:
                continue

            color: int = self.board_state[point]
            if color == EMPTY:
                visited[point] = True
                continue

            members: np.ndarray = np.zeros_like(self.board_state, dtype=bool)
            neighbors: np.ndarray = np.zeros_like(self.board_state, dtype=bool)

            self._flow(point, color, visited, members, neighbors)

            captain: int = int(np.argmax(members))
            new_group: Group = Group(color, captain, members, neighbors, self)

            for member_idx in np.where(members)[0]:
                self.group_index[member_idx] = group_count

            self.groups.append(new_group)
            group_count += 1

        self._compute_neighbor_relationships()

    def _flow(
        self,
        point: int,
        color: int,
        visited: np.ndarray,
        members: np.ndarray,
        neighbors: np.ndarray,
    ) -> None:
        if visited[point]:
            return

        visited[point] = True
        members[point] = True

        neighbor_points: np.ndarray = get_neighbors(point)

        for neighbor_point in neighbor_points:
            if visited[neighbor_point]:
                continue

            neighbor_color: int = int(self.board_state[neighbor_point])

            if neighbor_color == color:
                self._flow(neighbor_point, color, visited, members, neighbors)
            else:
                neighbors[neighbor_point] = True

    def _compute_neighbor_relationships(self) -> None:
        for group in self.groups:
            neighbor_group_set: Set[int] = set()
            for neighbor_point in np.where(group.neighbors)[0]:
                neighbor_group_idx: int = int(self.group_index[neighbor_point])
                if neighbor_group_idx != self.group_index[group.captain]:
                    neighbor_group_set.add(neighbor_group_idx)
            group.neighbor_indices = list(neighbor_group_set)
            group.compute_colorset_neighbors()

    def num_groups(self, color: int | None = None) -> int:
        if color is None:
            return len(self.groups)
        return sum(1 for group in self.groups if group.color == color)

    def get_group(self, point_index: int) -> Group:
        return self.groups[int(self.group_index[point_index])]

    def captain_of(self, point_index: int) -> int:
        return self.get_group(point_index).captain

    def is_captain(self, point_index: int) -> bool:
        return point_index == self.captain_of(point_index)

    def neighbors(self, point_index: int) -> np.ndarray:
        return self.get_group(point_index).neighbors

    def neighbors_by_color(self, point_index: int, color: int) -> np.ndarray:
        return self.get_group(point_index).neighbors_for_colorset({color})

    def captainize_bitset(self, points_mask: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.zeros(len(self.groups), dtype=bool)
        for point_idx in np.where(points_mask)[0]:
            group_idx: int = int(self.group_index[point_idx])
            result[group_idx] = True
        return result

    def is_game_over(self) -> bool:
        return self.get_winner() != EMPTY

    def get_winner(self) -> int:
        for color in [BLUE, RED]:
            if self._check_connection(color):
                return color
        return EMPTY

    def _check_connection(self, color: int) -> bool:
        for group in self.groups:
            if group.color != color:
                continue

            top_connected: bool = False
            bottom_connected: bool = False

            for member_idx in np.where(group.members)[0]:
                row: int = member_idx // BOARD_SIZE
                if row == 0:
                    top_connected = True
                if row == BOARD_SIZE - 1:
                    bottom_connected = True

            if top_connected and bottom_connected:
                return True

        return False


class GroupBuilder:
    @staticmethod
    def build(board_state: np.ndarray) -> Groups:
        groups = Groups(board_state)
        return groups