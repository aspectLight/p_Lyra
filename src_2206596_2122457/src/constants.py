# src/constants.py
"""
Game Constants
==============
Centralized constants for the game system including board configuration and piece types.
"""

# ==================== BOARD CONFIGURATION ====================
BOARD_SIZE = 14
TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE

# ==================== CLUSTER CONSTANTS ====================
CLUSTER_EMPTY = -1
PLAYER_NONE = 0
PLAYER_B = 1
PLAYER_R = 2

# ==================== PIECE TYPES ====================
# Basic piece types
PIECE_EMPTY = 0
PIECE_B = 1
PIECE_R = 2

# Edge piece types (pieces touching board edges)
PIECE_B_EDGE = 3
PIECE_R_EDGE = 4

# Invalid piece type (for coordinates outside the board)
PIECE_INVALID = 5

# ==================== CONVENIENCE SETS ====================
PIECE_OCCUPIED = {PIECE_B, PIECE_R}
PIECE_BLUE = {PIECE_B, PIECE_B_EDGE}
PIECE_RED = {PIECE_R, PIECE_R_EDGE}
PIECE_EDGE = {PIECE_B_EDGE, PIECE_R_EDGE}

# ==================== NEIGHBOR LOOKUP ====================
NEIGHBOR_PADDING = -1

# ==================== Time Constants ====================
TIME_MAX = 1000
