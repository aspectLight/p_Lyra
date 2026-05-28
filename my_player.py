from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from typing import cast, Any
from game_state_hex import GameStateHex
from src_2206596_2122457.src.mcts.mcts import MCTS
from src_2206596_2122457.src.config import MCTSConfig
from src_2206596_2122457.src.game_trackers.player_identity import PlayerIdentity

class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer", *args: Any, **kwargs: Any) -> None: 
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name, *args, **kwargs)  # type: ignore[misc]
        self._mcts = MCTS(MCTSConfig())
        self._player_identity: PlayerIdentity | None = None

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs: Any) -> Action:  # type: ignore[override]
        """
        Compute the action using MCTS with the refactored APIs.

        Args:
            current_state (GameState): The current game state (Hex-specific state expected).

        Returns:
            Action: A LightAction containing our piece and the selected position.
        """
        self._initialize_mcts_if_needed(cast(GameStateHex, current_state))

        # _player_identity is guaranteed initialized by _initialize_mcts_if_needed
        best_position = self._mcts.get_action(cast(GameStateHex, current_state), cast(PlayerIdentity, self._player_identity))
        return LightAction({
            "position": best_position,
            "piece": self.get_piece_type(),
        })

    def _initialize_mcts_if_needed(self, state: GameStateHex) -> None:
        if self._player_identity is None:
            our_player_id = state.get_next_player().get_id()
            our_piece_type = self.get_piece_type()
            self._player_identity = PlayerIdentity(our_player_id, our_piece_type)

