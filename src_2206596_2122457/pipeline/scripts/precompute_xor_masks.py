#!/usr/bin/env python3
# pipeline/scripts/precompute_xor_masks.py

import os
import sys
import numpy as np

# Ensure project root on sys.path to import pipeline.out modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Use the builder output so we stay consistent with runtime tables
from pipeline.out.local_pattern_table import OFFSET_TABLE, OFFSET_LIST  # type: ignore

OUTPUT_DIR = os.path.join("pipeline", "out")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "xor_masks.py")


def build_xor_masks(offset_table: dict[int, list[int]], num_offsets: int) -> np.ndarray:
    max_piece_code = max(offset_table.keys())
    masks = np.zeros((max_piece_code + 1, num_offsets), dtype=np.uint64)
    for p, arr in offset_table.items():
        # Ensure length matches num_offsets; pad or trim as needed
        if len(arr) < num_offsets:
            data = arr + [0] * (num_offsets - len(arr))
        else:
            data = arr[:num_offsets]
        masks[p] = np.array(data, dtype=np.uint64)
    return masks


def write_output_py(filepath: str, masks: np.ndarray) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Auto-generated XOR mask table. Do not edit.\n")
        f.write("import numpy as np\n\n")
        f.write("XOR_MASKS = np.array(\n")
        f.write(repr(masks.tolist()))
        f.write(", dtype=np.uint64)\n")
    print(f"✅ Wrote XOR masks to {filepath}")


if __name__ == "__main__":
    num_offsets = len(OFFSET_LIST)
    masks = build_xor_masks(OFFSET_TABLE, num_offsets)
    write_output_py(OUTPUT_FILE, masks)


