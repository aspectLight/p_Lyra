import numpy as np
from src_2206596_2122457.src.constants import PLAYER_NONE, PLAYER_B, PLAYER_R

class Clusters:
    __slots__ = (
        'board_size',
        'total_cells',
        '_neighbors_array',
        '_edge_flags',
        '_cell_to_parent',
        '_cell_to_player',
        '_winner',
        '_north_idx',
        '_south_idx',
        '_west_idx',
        '_east_idx',
    )

    def __init__(self, board_size: int, neighbors_array: np.ndarray, edge_flags: np.ndarray) -> None:
        self.board_size = board_size
        self.total_cells = neighbors_array.shape[0]

        self._neighbors_array = (
            neighbors_array if neighbors_array.dtype == np.int32 else neighbors_array.astype(np.int32, copy=False)
        )
        self._edge_flags = (
            edge_flags if edge_flags.dtype == np.bool_ else edge_flags.astype(np.bool_, copy=False)
        )

        size_with_edges = self.total_cells + 4
        self._cell_to_parent = np.arange(size_with_edges, dtype=np.int32)
        self._cell_to_player = np.zeros(size_with_edges, dtype=np.int8)

        self._north_idx = self.total_cells + 0
        self._south_idx = self.total_cells + 1
        self._west_idx = self.total_cells + 2
        self._east_idx = self.total_cells + 3

        self._winner = None

    def build_from_board_state(self, board_state: np.ndarray) -> None:
        self._cell_to_parent[:] = np.arange(self.total_cells + 4, dtype=np.int32)
        self._cell_to_player[:] = 0
        self._winner = None

        self._cell_to_player[self._north_idx:self._east_idx + 1] = [PLAYER_R, PLAYER_R, PLAYER_B, PLAYER_B]

        occupied = np.flatnonzero(board_state)
        for cell_idx in occupied:
            player = int(board_state[cell_idx])
            if player != PLAYER_NONE:
                self.add_piece_and_update_groups(cell_idx, player)

    def add_piece_and_update_groups(self, cell_idx: int, player: int) -> int:
        ctp = self._cell_to_player
        ctp[cell_idx] = player

        neighbors = self._neighbors_array[cell_idx]
        for n in neighbors:
            if n >= 0 and ctp[n] == player:
                self._union(cell_idx, n)

        flags = self._edge_flags[cell_idx]
        if player == PLAYER_B:
            if flags[2]: self._union(cell_idx, self._west_idx)
            if flags[3]: self._union(cell_idx, self._east_idx)
            if self._find(self._west_idx) == self._find(self._east_idx):
                self._winner = PLAYER_B
        else:
            if flags[0]: self._union(cell_idx, self._north_idx)
            if flags[1]: self._union(cell_idx, self._south_idx)
            if self._find(self._north_idx) == self._find(self._south_idx):
                self._winner = PLAYER_R

        return int(self._find(cell_idx))

    def _find(self, x: int) -> int:
        parent = self._cell_to_parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(self, x: int, y: int) -> None:
        parent = self._cell_to_parent
        rx = self._find(x)
        ry = self._find(y)
        if rx != ry:
            parent[rx] = ry

    def get_winner(self) -> int | None:
        return self._winner

    def copy(self) -> 'Clusters':
        new = Clusters(self.board_size, self._neighbors_array, self._edge_flags)
        new._cell_to_parent = self._cell_to_parent.copy()
        new._cell_to_player = self._cell_to_player.copy()
        new._winner = self._winner
        return new
