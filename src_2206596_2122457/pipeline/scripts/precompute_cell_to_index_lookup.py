#!/usr/bin/env python3
# pipeline/scripts/precompute_cell_to_index_lookup.py

"""
Precompute Cell to Index Lookup
--------------------------------
Generates a 2D NumPy array for O(1) (q, r) to cell_index lookups on hex boards.

Outputs:
    pipeline/out/cell_to_index_lookup.py
"""

import os
import numpy as np

BOARD_SIZE = 14
OUTPUT_DIR = "pipeline/out"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cell_to_index_lookup.py")


def build_cell_to_index_array(board_size: int) -> np.ndarray:
    """Build 2D lookup array mapping (q, r) coordinates to cell indices.
    
    Returns:
        cell_to_index: shape (board_size, board_size) with -1 for invalid cells
    """
    cell_to_index = np.full((board_size, board_size), -1, dtype=np.int32)
    
    cell_idx = 0
    for q in range(board_size):
        for r in range(board_size):
            cell_to_index[q, r] = cell_idx
            cell_idx += 1
    
    return cell_to_index


def write_cell_to_index_lookup(lookup: np.ndarray, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("# ======================================================\n")
        f.write("# Auto-generated Cell to Index Lookup - DO NOT EDIT\n")
        f.write("# ======================================================\n")
        f.write("# 2D NumPy array for O(1) (q, r) to cell_index lookups\n")
        f.write("# Shape: (board_size, board_size)\n")
        f.write("# Usage: cell_idx = CELL_TO_INDEX_LOOKUP[q, r]\n")
        f.write("# ======================================================\n\n")
        
        f.write("import numpy as np\n\n")
        
        f.write(f"CELL_TO_INDEX_LOOKUP = np.array(\n")
        f.write(repr(lookup.tolist()))
        f.write(", dtype=np.int32)\n")
    
    print(f"✅ Cell to index lookup written to {output_file}")
    print(f"   - Shape: {lookup.shape}")
    print(f"   - Total cells: {lookup.size}")


if __name__ == "__main__":
    print(f"🔧 Precomputing cell to index lookup for {BOARD_SIZE}x{BOARD_SIZE} board...")
    
    lookup = build_cell_to_index_array(BOARD_SIZE)
    write_cell_to_index_lookup(lookup, OUTPUT_FILE)
    
    print("✨ Done!")