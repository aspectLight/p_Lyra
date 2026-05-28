
from typing import Tuple, List, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from src_2206596_2122457.src.constants import BOARD_SIZE, PIECE_EMPTY, PIECE_B, PIECE_R

if TYPE_CHECKING:
    from src_2206596_2122457.src.pattern_system.field_propagation_pattern_store import FieldPropagationPatternStore

def action_to_id(position: Tuple[int, int]) -> int:
    return position[0] * BOARD_SIZE + position[1]

def board_to_ascii(board: NDArray[np.int_]) -> str:
    b: NDArray[np.int8] = np.asarray(board, dtype=np.int8)
    rows: List[str] = []
    for q in range(BOARD_SIZE):
        row_vals: NDArray[np.int8] = b[q * BOARD_SIZE:(q + 1) * BOARD_SIZE]
        chars: List[str] = []
        for v in row_vals:
            if int(v) == PIECE_EMPTY:
                chars.append(".")
            elif int(v) == PIECE_B:
                chars.append("B")
            elif int(v) == PIECE_R:
                chars.append("R")
            else:
                chars.append(str(int(v)))
        rows.append(" ".join(chars))
    return "\n".join(rows)

def board_to_hex_ascii(board: NDArray[np.int_], last_move: Optional[int] = None, use_color: bool = True) -> str:
    b: NDArray[np.int8] = np.asarray(board, dtype=np.int8)
    rows: List[str] = []
    COLOR_R = "\x1b[31m" if use_color else ""
    COLOR_B = "\x1b[34m" if use_color else ""
    COLOR_DIM = "\x1b[2m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""
    for q in range(BOARD_SIZE):
        indent = " " * q
        parts: List[str] = []
        for r in range(BOARD_SIZE):
            idx = q * BOARD_SIZE + r
            v = int(b[idx])
            is_last = (last_move == idx)
            if v == PIECE_EMPTY:
                sym = f"{COLOR_DIM}.{RESET}"
            elif v == PIECE_B:
                sym = f"{COLOR_B}B{RESET}"
            elif v == PIECE_R:
                sym = f"{COLOR_R}R{RESET}"
            else:
                sym = str(v)
            if is_last:
                sym = f"({sym})"
            else:
                sym = f" {sym} "
            parts.append(sym)
        rows.append(indent + "".join(parts))
    return "\n".join(rows)

def board_with_pattern_table(
    board: NDArray[np.int_],
    last_move: Optional[int],
    store: "FieldPropagationPatternStore",
    player: int,
    global_table: Dict[int, Dict[int, Dict[str, Any]]],
    local_table: Dict[int, Dict[int, Dict[str, Any]]],
    last_move_display: Optional[int] = None,
    use_color: bool = False,
) -> str:
    from src.constants import PIECE_EMPTY
    board_lines: List[str] = board_to_hex_ascii(board, last_move_display, use_color).split("\n")
    if last_move is None:
        return "\n".join(board_lines)
    affected: List[int] = store.cells_affected_by.get(int(last_move), [])
    pattern_rows: List[str] = []
    pattern_rows.append("Pattern Table (last=" + str(last_move) + "):")
    pattern_rows.append("Cell | Key6      | Key12      | Gamma(G)  | Gamma(L)  ")
    pattern_rows.append("-" * 60)
    for ai in affected:
        aid: int = int(ai)
        if int(board[aid]) != PIECE_EMPTY:
            continue
        k6: int
        k12: int
        k6, k12 = store.get_keys(aid)
        gp: Optional[Dict[str, Any]] = global_table.get(player, {}).get(int(k12)) or global_table.get(player, {}).get(int(k6))
        lp: Optional[Dict[str, Any]] = local_table.get(player, {}).get(int(k12)) or local_table.get(player, {}).get(int(k6))
        g_gamma: Optional[float] = gp.get("gamma") if gp else None
        l_gamma: Optional[float] = lp.get("gamma") if lp else None
        g_str: str = f"{g_gamma:.6f}" if g_gamma is not None else "None    "
        l_str: str = f"{l_gamma:.6f}" if l_gamma is not None else "None    "
        pattern_rows.append(f"{aid:4d} | {int(k6):10d} | {int(k12):10d} | {g_str:9s} | {l_str:9s}")
    max_board_width: int = max(len(line.rstrip()) for line in board_lines) if board_lines else 0
    max_board_lines: int = len(board_lines)
    max_pattern_lines: int = len(pattern_rows)
    max_lines: int = max(max_board_lines, max_pattern_lines)
    combined: List[str] = []
    padding_width: int = max(70, max_board_width + 10)
    for i in range(max_lines):
        board_part: str = board_lines[i] if i < max_board_lines else ""
        pattern_part: str = pattern_rows[i] if i < max_pattern_lines else ""
        board_stripped: str = board_part.rstrip()
        padding: str = " " * max(0, padding_width - len(board_stripped))
        combined.append(board_part + padding + pattern_part)
    return "\n".join(combined)