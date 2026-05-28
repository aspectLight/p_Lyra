#!/usr/bin/env python3
# pipeline/scripts/pattern_builder.py - FIXED VERSION

import os
import random
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
import numpy as np

# ==================== MODE SWITCH ====================
GAMMA_MODE: str = "global"  # "global" | "local"

OUTPUT_DIR = "pipeline/out"
INPUT_FILE = (
    "pipeline/data/global_patterns.txt" if GAMMA_MODE == "global" else "pipeline/data/local_patterns.txt" # type: ignore
)
OUTPUT_NAME = (
    "global_pattern_table" if GAMMA_MODE == "global" else "local_pattern_table" # type: ignore
)

SEED: int = 0
BOARD_SIZE: int = 14
INCLUDE_PATTERNS: bool = True
GLOBAL_GAMMA_CAP: float = 0.157

# Numeric piece constants
PIECE_EMPTY = 0
PIECE_B = 1
PIECE_R = 2
PIECE_B_EDGE = 3
PIECE_R_EDGE = 4
PIECE_PADDING = 0

PIECE_MAP: Dict[int, int] = {
    PIECE_EMPTY: PIECE_EMPTY,
    PIECE_B: PIECE_B,
    PIECE_R: PIECE_R,
    PIECE_B_EDGE: PIECE_B_EDGE,
    PIECE_R_EDGE: PIECE_R_EDGE,
    PIECE_PADDING: PIECE_PADDING,
}

HEX_DIRECTIONS: List[Tuple[int, int]] = [
    (+1, 0), (+1, -1), (0, -1),
    (-1, 0), (-1, +1), (0, +1),
]

# ==================== FIXED PERMUTATION TABLES ====================
# YOUR INDEX_12 LAYOUT:
# - Indices 0-5: Ring 1 (positions 1-6, adjacent to center)
# - Indices 6-11: Ring 2 (positions 7-12, further from center)
#
# EACH RING ROTATES INDEPENDENTLY!
# ROT_TABLE[new_idx] = old_idx means "new position gets value from old position"

ROT_TABLE = {
    # 6 elements: just Ring 1 rotating
    6: [5, 0, 1, 2, 3, 4],
    
    # 12 elements: Ring 1 and Ring 2 rotate independently
    12: [
        # Ring 1 (indices 0-5): rotates within itself
        5, 0, 1, 2, 3, 4,
        # Ring 2 (indices 6-11): rotates within itself
        11, 6, 7, 8, 9, 10
    ]
}

# Piece flip mapping for mirroring: swap 1<->2 (blue<->red), 3<->4 (edges)
PIECE_FLIP = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}

# Mirror mapping: reflects across vertical axis
# EACH RING MIRRORS INDEPENDENTLY!
MIRROR_TABLE = {
    # 6 elements: just Ring 1 mirroring
    6: [0, 5, 4, 3, 2, 1],
    
    # 12 elements: Ring 1 and Ring 2 mirror independently
    12: [
        # Ring 1 (indices 0-5): mirrors across axis
        0, 5, 4, 3, 2, 1,
        # Ring 2 (indices 6-11): mirrors across axis
        6, 11, 10, 9, 8, 7
    ]
}

def build_cell_indices(board_size: int) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    coords: List[Tuple[int, int]] = []
    cell_to_index: Dict[Tuple[int, int], int] = {}
    for q in range(board_size):
        for r in range(board_size):
            idx = len(coords)
            coord: Tuple[int, int] = (q, r)
            coords.append(coord)
            cell_to_index[coord] = idx
    index_to_cell: np.ndarray = np.array(coords, dtype=np.int32)
    return index_to_cell, cell_to_index

# MoHex neighbor ordering (center at index 0, then ring-1, ring-2)
MOHEX_OFFSET_LIST_12: List[Tuple[int, int]] = [
    (0, 0),
    (-1, 0), (0, 1), (-1, -1), (1, 1), (0, -1), (1, 0),
    (-2, 0), (-1, 1), (0, 2), (1, 2), (2, 1), (2, 0),
]

def build_local_neighbor_map(board_size: int, offset_list: List[Tuple[int, int]], 
                             cell_to_index: Dict[Tuple[int, int], int]) -> List[np.ndarray]:
    neighbor_map: List[np.ndarray] = []
    
    for q in range(board_size):
        for r in range(board_size):
            neighbors: List[int] = []
            
            for dq, dr in offset_list:
                nq = q + dq
                nr = r + dr
                
                if 0 <= nq < board_size and 0 <= nr < board_size:
                    neighbor_idx = cell_to_index[(nq, nr)]
                    neighbors.append(neighbor_idx)
            
            neighbor_map.append(np.array(neighbors, dtype=np.int32))
    
    return neighbor_map

def build_offset_lookup(offset_list: List[Tuple[int, int]], radius: int) -> Tuple[np.ndarray, int]:
    min_d: int = -radius
    size: int = 2 * radius + 1
    lookup: np.ndarray = np.full((size, size), -1, dtype=np.int32)
    
    for idx, (dq, dr) in enumerate(offset_list):
        lookup[dq - min_d, dr - min_d] = idx
    
    return lookup, min_d

def generate_offset_table_arrays(seed: int, num_offsets: int) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    table: Dict[int, List[int]] = {}
    for piece_code in [PIECE_EMPTY, PIECE_B, PIECE_R, PIECE_B_EDGE, PIECE_R_EDGE]:
        if piece_code in (PIECE_EMPTY, PIECE_PADDING):
            table[piece_code] = [0] * num_offsets
        else:
            if piece_code not in table:
                table[piece_code] = [rng.getrandbits(64) for _ in range(num_offsets)]
    return table

def initial_board_state_array(board_size: int) -> List[int]:
    return [PIECE_EMPTY] * (board_size * board_size)

def ring_coords(radius: int) -> List[Tuple[int, int]]:
    if radius == 0:
        return []
    coords: List[Tuple[int, int]] = []
    q, r = -radius, 0
    for direction_idx in range(6):
        dq, dr = HEX_DIRECTIONS[direction_idx]
        for _ in range(radius):
            coords.append((q, r))
            q += dq
            r += dr
    return coords

def normalize_pattern_str(s: str) -> str:
    return "".join(ch for ch in s.strip() if ch.isdigit())

def rotate_pattern_inplace(pattern: List[int]) -> None:
    size = len(pattern)
    if size not in ROT_TABLE:
        return
    
    rot = ROT_TABLE[size]
    temp = pattern[:]
    
    for new_idx, old_idx in enumerate(rot):
        pattern[new_idx] = temp[old_idx]

def mirror_pattern_inplace(pattern: List[int]) -> None:
    size = len(pattern)
    
    for i in range(size):
        pattern[i] = PIECE_FLIP.get(pattern[i], pattern[i])
    
    if size in MIRROR_TABLE:
        mirror = MIRROR_TABLE[size]
        temp = pattern[:]
        for new_idx, old_idx in enumerate(mirror):
            pattern[new_idx] = temp[old_idx]

def generate_pattern_variants(pattern: List[int]) -> Set[Tuple[int, ...]]:
    variants: Set[Tuple[int, ...]] = set()
    
    current = pattern[:]
    for _ in range(6):
        variants.add(tuple(current))
        rotate_pattern_inplace(current)
    
    mirrored = pattern[:]
    mirror_pattern_inplace(mirrored)
    current = mirrored
    for _ in range(6):
        variants.add(tuple(current))
        rotate_pattern_inplace(current)
    
    return variants

def playout_global_gamma(type_: int, gamma: float, global_gamma_cap: float) -> float:
    if type_ == 0:
        return min(gamma, global_gamma_cap)
    elif type_ in (1, 2):
        return 0.00001
    elif type_ == 3:
        return 0.0001
    return gamma

def playout_local_gamma(type_: int, gamma: float) -> float:
    if type_ == 0:
        return gamma
    elif type_ in (1, 2):
        return 0.00001
    elif type_ == 3:
        return 0.0001
    return gamma

class PatternKeyComputer:
    
    def __init__(self, offset_list: List[Tuple[int, int]], offset_table: Dict[int, List[int]]) -> None:
        self.offset_list: List[Tuple[int, int]] = offset_list
        self.offset_table: Dict[int, List[int]] = offset_table
        self.offset_index: Dict[Tuple[int, int], int] = {offset_list[i]: i for i in range(len(offset_list))}

    def _map_val_to_piece(self, v: int, player: int) -> int:
        if v == 0:
            return PIECE_EMPTY
        if v == 1:
            return player
        if v == 2:
            return PIECE_R if player == PIECE_B else PIECE_B
        if v == 3:
            return PIECE_B_EDGE if player == PIECE_B else PIECE_R_EDGE
        if v == 4:
            return PIECE_R_EDGE if player == PIECE_B else PIECE_B_EDGE
        return PIECE_EMPTY

    def compute_key(self, pattern: List[int], player: int, pattern_offsets: List[Tuple[int, int]]) -> int:
        key: int = 0
        for i, val in enumerate(pattern):
            if i >= len(pattern_offsets):
                break
            offset = pattern_offsets[i]
            idx = self.offset_index.get(offset)
            if idx is None:
                continue
            piece_code = self._map_val_to_piece(val, player)
            key ^= self.offset_table[piece_code][idx]
        return key

def load_patterns(filepath: str) -> List[Tuple[str, float, int]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    out: List[Tuple[str, float, int]] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("gamma"):
                continue
            parts: List[str] = line.split()
            if len(parts) < 3:
                continue
            pat, gamma_str, type_str = parts[0], parts[1], parts[2]
            try:
                gamma: float = float(gamma_str)
                type_: int = int(type_str)
            except ValueError:
                continue
            out.append((pat, gamma, type_))
    return out

def write_output_py(filepath: str,
                    seed: int,
                    board_size: int,
                    index_to_cell: np.ndarray,
                    cell_to_index: Dict[Tuple[int, int], int],
                    neighbor_map: List[np.ndarray],
                    offset_list: List[Tuple[int, int]],
                    offset_lookup: np.ndarray,
                    min_d: int,
                    offset_table: Dict[int, List[int]],
                    initial_board: List[int],
                    pattern_table: Dict[int, Dict[int, Dict[str, object]]]) -> None:
    with open(filepath, "w") as f:
        f.write("# Auto-generated by pipeline/scripts/pattern_builder.py\n")
        f.write("import numpy as np\n\n")
        f.write(f"SEED = {seed}\n\n")
        
        f.write("INDEX_TO_CELL = np.array(\n")
        f.write(repr(index_to_cell.tolist()))
        f.write(", dtype=np.int32)\n\n")
        
        f.write("CELL_TO_INDEX = " + repr(cell_to_index) + "\n\n")
        f.write("NEIGHBOR_MAP = [")
        for i, arr in enumerate(neighbor_map):
            if i > 0:
                f.write(", ")
            f.write("np.array(")
            f.write(repr(arr.tolist()))
            f.write(", dtype=np.int32)")
        f.write("]\n\n")
        
        f.write("OFFSET_LIST = " + repr(offset_list) + "\n\n")
        f.write("OFFSET_LOOKUP = np.array(\n")
        f.write(repr(offset_lookup.tolist()))
        f.write(", dtype=np.int32)\n")
        f.write(f"OFFSET_LOOKUP_MIN_D = {min_d}\n\n")
        
        f.write("OFFSET_TABLE = " + repr(offset_table) + "\n\n")
        f.write("INITIAL_BOARD_STATE = " + repr(initial_board) + "\n\n")
        f.write("PATTERN_TABLE = " + repr(pattern_table) + "\n")
    print(f"✅ Wrote: {filepath}")

def build_all(input_file: str = INPUT_FILE, 
              output_dir: str = OUTPUT_DIR, 
              output_name: str = OUTPUT_NAME,
              seed: int = SEED, 
              board_size: int = BOARD_SIZE, 
              include_patterns: bool = INCLUDE_PATTERNS) -> str:
    os.makedirs(output_dir, exist_ok=True)
    patterns: List[Tuple[str, float, int]] = load_patterns(input_file)
    print(f"✅ Loaded {len(patterns)} patterns from '{input_file}'")

    index_to_cell, cell_to_index = build_cell_indices(board_size)
    print(f"✅ Built cell indexing: {len(index_to_cell)} cells")

    offset_list: List[Tuple[int, int]] = MOHEX_OFFSET_LIST_12
    radius = 2
    offset_lookup, min_d = build_offset_lookup(offset_list, radius)
    offset_table: Dict[int, List[int]] = generate_offset_table_arrays(seed, len(offset_list))
    print(f"✅ Built offset lookup: {len(offset_list)} offsets, min_d={min_d}")

    neighbor_map: List[np.ndarray] = build_local_neighbor_map(board_size, offset_list, cell_to_index)
    initial_board: List[int] = initial_board_state_array(board_size)

    pkc: PatternKeyComputer = PatternKeyComputer(offset_list, offset_table)

    pattern_table: Dict[int, Dict[int, Dict[str, object]]] = {PIECE_B: {}, PIECE_R: {}}
    print(f"🔄 Processing patterns for mode: {GAMMA_MODE}...")
    
    unique_variants: int = 0
    for pattern_str, gamma, type_ in tqdm(patterns, desc="Patterns"):
        digits: str = normalize_pattern_str(pattern_str)
        if not digits:
            continue
        base: List[int] = [int(c) for c in digits]

        n: int = len(base)
        pattern_offsets = offset_list[1:n+1]
        
        variants = generate_pattern_variants(base)
        unique_variants += len(variants)

        for var in variants:
            bkey: int = pkc.compute_key(list(var), PIECE_B, pattern_offsets)
            rkey: int = pkc.compute_key(list(var), PIECE_R, pattern_offsets)

            if GAMMA_MODE == "global":
                gamma_value: float = playout_global_gamma(type_, gamma, GLOBAL_GAMMA_CAP)
            else:
                gamma_value = playout_local_gamma(type_, gamma)

            entry: Dict[str, object] = {"gamma": gamma_value}
            if include_patterns:
                entry["pattern"] = list(var)

            pattern_table[PIECE_B][bkey] = entry
            pattern_table[PIECE_R][rkey] = entry

    out_path: str = os.path.join(output_dir, f"{output_name}.py")
    write_output_py(out_path, seed, board_size, index_to_cell, cell_to_index,
                    neighbor_map, offset_list, offset_lookup, min_d,
                    offset_table, initial_board, pattern_table)

    print("\n✅ Build complete.")
    print(f"   Mode: {GAMMA_MODE}")
    print(f"   Input patterns: {len(patterns)}")
    print(f"   Unique variants generated: {unique_variants}")
    print(f"   Pattern entries (B): {len(pattern_table[PIECE_B])}")
    print(f"   Pattern entries (R): {len(pattern_table[PIECE_R])}")
    return out_path

if __name__ == "__main__":
    build_all()