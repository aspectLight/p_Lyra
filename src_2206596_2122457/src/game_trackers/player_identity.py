from typing import Any, List


class PlayerIdentity:
    def __init__(self, player_id: int, piece_type: str) -> None:
        """
        Initialize PlayerIdentity with a player ID and piece type string.
        
        Args:
            player_id (int): The player's ID.
            piece_type (str): The piece type as a string ("R" or "B") from the external implementation.
        """
        self._our_player_id = player_id
        self._our_piece_type = 2 if piece_type == "R" else 1  # PIECE_R=2, PIECE_B=1
        self._opponent_piece_type = 1 if piece_type == "R" else 2
        self._our_piece_letter = piece_type
        self._opponent_piece_letter = "B" if piece_type == "R" else "R"
    
    def get_our_player_id(self) -> int:
        return self._our_player_id
    
    def get_our_piece_type(self) -> int:
        """
        Get our piece type as an integer.
        
        Returns:
            int: 1 for PIECE_B, 2 for PIECE_R
        """
        return self._our_piece_type
    
    def get_opponent_piece_type(self) -> int:
        """
        Get opponent's piece type as an integer.
        
        Returns:
            int: 1 for PIECE_B, 2 for PIECE_R
        """
        return self._opponent_piece_type

    def get_our_piece_letter(self) -> str:
        return self._our_piece_letter

    def get_opponent_piece_letter(self) -> str:
        return self._opponent_piece_letter
    
    def is_our_piece(self, piece_type: int) -> bool:
        """
        Check if a piece type (as integer) belongs to us.
        
        Args:
            piece_type (int): The piece type to check.
            
        Returns:
            bool: True if it's our piece type.
        """
        return piece_type == self._our_piece_type
    
    def is_opponent_piece(self, piece_type: int) -> bool:
        """
        Check if a piece type (as integer) belongs to the opponent.
        
        Args:
            piece_type (int): The piece type to check.
            
        Returns:
            bool: True if it's the opponent's piece type.
        """
        return piece_type == self._opponent_piece_type
    
    def get_our_player_index(self, players: List[Any]) -> int:
        return 0 if self._our_player_id == players[0].get_id() else 1
    
    def get_opponent_player_index(self, players: List[Any]) -> int:
        return 1 - self.get_our_player_index(players)