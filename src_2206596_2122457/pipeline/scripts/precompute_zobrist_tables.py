#!/usr/bin/env python3
# pipeline/scripts/precompute_zobrist_tables.py

"""
Precompute Zobrist Hash Tables
------------------------------
Generates deterministic Zobrist hashing tables and player bits
for fast board hashing in the game engine.

Outputs:
    pipeline/out/zobrist_constants.py
"""

import os
import random
import numpy as np

# ======================================================
# CONFIG
# ======================================================
BOARD_SIZE = 14
NUM_POSITIONS = BOARD_SIZE * BOARD_SIZE
SEED = 1337  # Change for different zobrist universes
OUTPUT_DIR = "pipeline/out"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "zobrist_constants.py")

# Piece codes (imported elsewhere in the engine)
PIECE_B = 1
PIECE_R = 2


def generate_zobrist_tables(board_size: int, seed: int):
    """Generate zobrist tables for each piece and player bits."""
    rnd = random.Random(seed)
    num_positions = board_size * board_size

    table_b = np.array([rnd.getrandbits(64) for _ in range(num_positions)], dtype=np.uint64)
    table_r = np.array([rnd.getrandbits(64) for _ in range(num_positions)], dtype=np.uint64)
    player_b = np.uint64(rnd.getrandbits(64))
    player_r = np.uint64(rnd.getrandbits(64))

    zobrist_tables = {
        PIECE_B: table_b,
        PIECE_R: table_r,
    }
    player_bits = {
        PIECE_B: player_b,
        PIECE_R: player_r,
    }

    return zobrist_tables, player_bits


def write_zobrist_constants(
    zobrist_tables: dict[int, np.ndarray],
    player_bits: dict[int, np.uint64],
    output_file: str,
    board_size: int,
    seed: int,
) -> None:
    """Write zobrist tables and player bits to a Python constants file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        f.write("# ======================================================\n")
        f.write("# Auto-generated Zobrist Constants - DO NOT EDIT\n")
        f.write("# ======================================================\n")
        f.write("# Precomputed 64-bit Zobrist hashing tables for pieces\n")
        f.write("# and player bits used in the game engine.\n")
        f.write(f"# Generated for BOARD_SIZE={board_size}, SEED={seed}\n")
        f.write("# ======================================================\n\n")

        f.write("import numpy as np\n\n")
        f.write(f"BOARD_SIZE = {board_size}\n")
        f.write(f"NUM_POSITIONS = {board_size * board_size}\n\n")
        f.write(f"PIECE_B = {PIECE_B}\n")
        f.write(f"PIECE_R = {PIECE_R}\n\n")

        # Write zobrist tables
        for piece, table in zobrist_tables.items():
            name = "ZOBRIST_TABLE_B" if piece == PIECE_B else "ZOBRIST_TABLE_R"
            f.write(f"{name} = np.array(\n")
            f.write(repr(table.tolist()))
            f.write(", dtype=np.uint64)\n\n")

        # Write player bits
        f.write(f"PLAYER_BIT_B = np.uint64({int(player_bits[PIECE_B])})\n")
        f.write(f"PLAYER_BIT_R = np.uint64({int(player_bits[PIECE_R])})\n\n")

        # Piece → zobrist_table and piece → player_bit dicts
        f.write("PIECE_TO_ZOBRIST_TABLE = {\n")
        f.write("    PIECE_B: ZOBRIST_TABLE_B,\n")
        f.write("    PIECE_R: ZOBRIST_TABLE_R,\n")
        f.write("}\n\n")

        f.write("PIECE_TO_PLAYER_BIT = {\n")
        f.write("    PIECE_B: PLAYER_BIT_B,\n")
        f.write("    PIECE_R: PLAYER_BIT_R,\n")
        f.write("}\n")

    print(f"✅ Zobrist constants written to {output_file}")
    print(f"   - Board size: {board_size}")
    print(f"   - Total positions: {board_size * board_size}")
    print(f"   - Seed: {seed}")


if __name__ == "__main__":
    print(f"🔧 Precomputing zobrist tables for {BOARD_SIZE}x{BOARD_SIZE} board (seed={SEED})...")

    zobrist_tables, player_bits = generate_zobrist_tables(BOARD_SIZE, SEED)
    write_zobrist_constants(zobrist_tables, player_bits, OUTPUT_FILE, BOARD_SIZE, SEED)

    print("✨ Done!")
