#!/usr/bin/env python3
"""
Precompute Matcher Tables
========================

Generates static NumPy arrays for pattern matching topology:
  - TABLE_SLICE_INDEX: Maps (center, neighbor) -> slice index
  - TABLE_GODEL_BITMASK: Maps (center, neighbor) -> godel bitmask
  - TABLE_EDGE_GODEL_BITMASK: Edge-specific godel bitmasks per slice
  - TABLE_INVERSE_CELL_LOOKUP: Reverse lookup from slice/bit -> neighbor cell

Outputs: pipeline/out/precomputed_matcher_tables.py
"""

import os
import sys
import numpy as np
from typing import Tuple

# ============================================================================
# CONSTANTS - Single Source of Truth
# ============================================================================

BOARD_SIZE = 14
PATTERN_MAX_RADIUS = 3
PATTERN_NUM_SLICES = 6
MAX_SLICE_BITS = PATTERN_MAX_RADIUS * (PATTERN_MAX_RADIUS + 1) // 2  # 6

OUTPUT_DIR = "pipeline/out"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "precomputed_matcher_tables.py")

# ============================================================================
# IMPORTS FROM PROJECT
# ============================================================================

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_neighbors_with_radii(board_size: int, max_radius: int) -> np.ndarray:
    """
    For each cell, compute neighbors organized by radius (1 to max_radius).
    
    Returns array of shape (total_cells, total_neighbor_slots) where:
      - Neighbors are grouped by increasing radius
      - Unfilled slots are -1
      - For radius r, there are at most r*(r+1)//2 positions
    
    Total slots = sum(r*(r+1)//2 for r in 1..max_radius) = max_radius*(max_radius+1)*(max_radius+2)//6
    """
    total_cells = board_size * board_size
    
    # Calculate total neighbor slots needed
    total_slots = sum(r * (r + 1) // 2 for r in range(1, max_radius + 1))
    
    neighbors_array = np.full((total_cells, total_slots), -1, dtype=np.int32)
    
    for center_idx in range(total_cells):
        q = center_idx // board_size
        r = center_idx % board_size
        
        slot_idx = 0
        
        for radius in range(1, max_radius + 1):
            # Generate all cells at this radius
            radius_neighbors: list[int] = []
            
            for dq in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    # Check if (dq, dr) is at the specified radius in hex space
                    if max(abs(dq), abs(dr), abs(dq + dr)) == radius:
                        nq, nr = q + dq, r + dr
                        if 0 <= nq < board_size and 0 <= nr < board_size:
                            neighbor_idx_val: int = nq * board_size + nr
                            radius_neighbors.append(neighbor_idx_val)
            
            # Place neighbors in slots
            for neighbor_idx_val in radius_neighbors:
                if slot_idx < total_slots:
                    neighbors_array[center_idx, slot_idx] = neighbor_idx_val
                    slot_idx += 1
    
    return neighbors_array


def compute_edge_topology(board_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify edge cells and assign edge indices.
    
    Edge indices:
      0: Top edge (r=0)
      1: Right edge (q=board_size-1)
      2: Bottom edge (r=board_size-1)
      3: Left edge (q=0)
    
    Returns:
      is_edge_cell: shape (num_cells,), bool, True if cell is on any edge
      edge_index_of_cell: shape (num_cells,), int32, edge index (0-3) or -1
    """
    num_cells = board_size * board_size
    is_edge_cell = np.zeros(num_cells, dtype=bool)
    edge_index_of_cell = np.full(num_cells, -1, dtype=np.int32)
    
    for cell_idx in range(num_cells):
        q = cell_idx // board_size
        r = cell_idx % board_size
        
        if r == 0:
            is_edge_cell[cell_idx] = True
            edge_index_of_cell[cell_idx] = 0
        elif q == board_size - 1:
            is_edge_cell[cell_idx] = True
            edge_index_of_cell[cell_idx] = 1
        elif r == board_size - 1:
            is_edge_cell[cell_idx] = True
            edge_index_of_cell[cell_idx] = 2
        elif q == 0:
            is_edge_cell[cell_idx] = True
            edge_index_of_cell[cell_idx] = 3
    
    return is_edge_cell, edge_index_of_cell


# ============================================================================
# MAIN PRECOMPUTATION
# ============================================================================

def precompute_matcher_tables(
    board_size: int,
    max_radius: int,
    num_slices: int,
    max_slice_bits: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute all matcher tables.
    
    Returns:
      table_slice_index: (num_cells, num_cells) int32
      table_godel_bitmask: (num_cells, num_cells) uint32
      table_edge_godel_bitmask: (num_cells, 4, num_slices) uint32
      table_inverse_cell_lookup: (num_cells, num_slices, max_slice_bits) int32
      is_edge_cell: (num_cells,) bool
      edge_index_of_cell: (num_cells,) int32
    """
    num_cells = board_size * board_size
    
    print(f"📊 Computing topology for {board_size}x{board_size} board ({num_cells} cells)...")
    
    # Compute edge topology
    is_edge_cell, edge_index_of_cell = compute_edge_topology(board_size)
    print(f"✓ Edge topology computed: {np.sum(is_edge_cell)} edge cells")
    
    # Compute neighbors organized by radius
    neighbors_by_radius = compute_neighbors_with_radii(board_size, max_radius)
    print(f"✓ Neighbors by radius: shape {neighbors_by_radius.shape}")
    
    # Initialize lookup tables
    table_slice_index = np.full((num_cells, num_cells), -1, dtype=np.int32)
    table_godel_bitmask = np.zeros((num_cells, num_cells), dtype=np.uint32)
    table_edge_godel_bitmask = np.zeros((num_cells, 4, num_slices), dtype=np.uint32)
    table_inverse_cell_lookup = np.full((num_cells, num_slices, max_slice_bits), -1, dtype=np.int32)
    
    print(f"📝 Initializing lookup tables...")
    
    # Fill lookup tables
    for center_idx in range(num_cells):
        slot_idx = 0
        
        for radius in range(1, max_radius + 1):
            slots_in_radius = radius * (radius + 1) // 2
            
            for bit_pos in range(slots_in_radius):
                neighbor_idx = neighbors_by_radius[center_idx, slot_idx]
                slot_idx += 1
                
                if neighbor_idx == -1:
                    continue
                
                # For each neighbor, it belongs to one slice at this radius
                slice_idx = radius - 1
                
                if slice_idx >= num_slices:
                    continue
                
                godel_bit = 1 << bit_pos
                
                # Record slice index mapping
                table_slice_index[center_idx, neighbor_idx] = slice_idx
                
                # Record godel bitmask
                table_godel_bitmask[center_idx, neighbor_idx] |= godel_bit
                
                # Record inverse lookup
                table_inverse_cell_lookup[center_idx, slice_idx, bit_pos] = neighbor_idx
                
                # If neighbor is edge cell, also record in edge godel bitmask
                if is_edge_cell[neighbor_idx]:
                    edge_idx = edge_index_of_cell[neighbor_idx]
                    if 0 <= edge_idx < 4:
                        table_edge_godel_bitmask[center_idx, edge_idx, slice_idx] |= godel_bit
    
    print(f"✓ Lookup tables populated")
    
    return (table_slice_index, table_godel_bitmask, table_edge_godel_bitmask, 
            table_inverse_cell_lookup, is_edge_cell, edge_index_of_cell)


def write_output_file(
    output_file: str,
    table_slice_index: np.ndarray,
    table_godel_bitmask: np.ndarray,
    table_edge_godel_bitmask: np.ndarray,
    table_inverse_cell_lookup: np.ndarray,
    is_edge_cell: np.ndarray,
    edge_index_of_cell: np.ndarray
) -> None:
    """Write all precomputed tables to output Python file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# ======================================================\n")
        f.write("# Auto-generated Matcher Tables - DO NOT EDIT\n")
        f.write("# ======================================================\n")
        f.write("# Precomputed topology tables for O(1) pattern matching\n")
        f.write("# ======================================================\n\n")
        
        f.write("import numpy as np\n\n")
        
        # Write TABLE_SLICE_INDEX
        f.write("# Maps (center_cell_idx, neighbor_idx) -> slice_index\n")
        f.write("# Value -1 indicates invalid mapping\n")
        f.write("# Shape: (num_cells, num_cells)\n")
        f.write("TABLE_SLICE_INDEX = np.array(\n")
        f.write(f"    {table_slice_index.tolist()!r},\n")
        f.write("    dtype=np.int32\n")
        f.write(")\n\n")
        
        # Write TABLE_GODEL_BITMASK
        f.write("# Maps (center_cell_idx, neighbor_idx) -> godel bitmask\n")
        f.write("# Bitmask indicates which bit position in the slice\n")
        f.write("# Shape: (num_cells, num_cells)\n")
        f.write("TABLE_GODEL_BITMASK = np.array(\n")
        f.write(f"    {table_godel_bitmask.tolist()!r},\n")
        f.write("    dtype=np.uint32\n")
        f.write(")\n\n")
        
        # Write TABLE_EDGE_GODEL_BITMASK
        f.write("# Edge-specific godel bitmasks\n")
        f.write("# Maps (center_cell_idx, edge_idx, slice_idx) -> godel bitmask\n")
        f.write("# Shape: (num_cells, 4, num_slices) where 4 edges are:\n")
        f.write("#   0: Top (r=0), 1: Right (q=max), 2: Bottom (r=max), 3: Left (q=0)\n")
        f.write("TABLE_EDGE_GODEL_BITMASK = np.array(\n")
        f.write(f"    {table_edge_godel_bitmask.tolist()!r},\n")
        f.write("    dtype=np.uint32\n")
        f.write(")\n\n")
        
        # Write TABLE_INVERSE_CELL_LOOKUP
        f.write("# Reverse lookup: (center_cell_idx, slice_idx, bit_pos) -> neighbor_idx\n")
        f.write("# Value -1 indicates no neighbor at this position\n")
        f.write("# Shape: (num_cells, num_slices, max_slice_bits)\n")
        f.write("TABLE_INVERSE_CELL_LOOKUP = np.array(\n")
        f.write(f"    {table_inverse_cell_lookup.tolist()!r},\n")
        f.write("    dtype=np.int32\n")
        f.write(")\n\n")
        
        # Write edge topology
        f.write("# Edge topology tables\n")
        f.write("IS_EDGE_CELL = np.array(\n")
        f.write(f"    {is_edge_cell.tolist()!r},\n")
        f.write("    dtype=bool\n")
        f.write(")\n\n")
        
        f.write("# Edge index for each cell (0-3 for edges, -1 for non-edge)\n")
        f.write("EDGE_INDEX_OF_CELL = np.array(\n")
        f.write(f"    {edge_index_of_cell.tolist()!r},\n")
        f.write("    dtype=np.int32\n")
        f.write(")\n\n")
        
        # Write metadata
        f.write("# Metadata\n")
        f.write(f"NUM_CELLS = {len(is_edge_cell)}\n")
        f.write(f"NUM_EDGE_CELLS = {np.sum(is_edge_cell)}\n")
        f.write(f"PATTERN_NUM_SLICES = 6\n")
        f.write(f"MAX_SLICE_BITS = 6\n")


def print_statistics(
    table_slice_index: np.ndarray,
    table_godel_bitmask: np.ndarray,
    table_edge_godel_bitmask: np.ndarray,
    table_inverse_cell_lookup: np.ndarray,
    is_edge_cell: np.ndarray
) -> None:
    """Print precomputation statistics."""
    print("\n" + "=" * 60)
    print("📊 PRECOMPUTATION STATISTICS")
    print("=" * 60)
    
    print(f"\n📍 Board Topology:")
    print(f"   Total cells: {len(is_edge_cell)}")
    print(f"   Edge cells: {np.sum(is_edge_cell)}")
    print(f"   Interior cells: {len(is_edge_cell) - np.sum(is_edge_cell)}")
    
    print(f"\n📊 Table Sizes:")
    print(f"   TABLE_SLICE_INDEX: {table_slice_index.shape} ({table_slice_index.nbytes / 1024:.2f} KB)")
    print(f"   TABLE_GODEL_BITMASK: {table_godel_bitmask.shape} ({table_godel_bitmask.nbytes / 1024:.2f} KB)")
    print(f"   TABLE_EDGE_GODEL_BITMASK: {table_edge_godel_bitmask.shape} ({table_edge_godel_bitmask.nbytes / 1024:.2f} KB)")
    print(f"   TABLE_INVERSE_CELL_LOOKUP: {table_inverse_cell_lookup.shape} ({table_inverse_cell_lookup.nbytes / 1024:.2f} KB)")
    
    total_size = (table_slice_index.nbytes + table_godel_bitmask.nbytes + 
                  table_edge_godel_bitmask.nbytes + table_inverse_cell_lookup.nbytes)
    print(f"\n   Total memory: {total_size / 1024:.2f} KB")
    
    # Coverage statistics
    valid_mappings = np.sum(table_slice_index >= 0)
    total_slots = np.prod(table_slice_index.shape)
    print(f"\n📈 Coverage:")
    print(f"   Valid slice mappings: {valid_mappings} / {total_slots} ({100*valid_mappings/total_slots:.1f}%)")
    print(f"   Non-zero godel masks: {np.sum(table_godel_bitmask > 0)} / {total_slots}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🔨 Precomputing matcher tables...\n")
    
    # Precompute all tables
    (table_slice_index, table_godel_bitmask, table_edge_godel_bitmask,
     table_inverse_cell_lookup, is_edge_cell, edge_index_of_cell) = \
        precompute_matcher_tables(BOARD_SIZE, PATTERN_MAX_RADIUS, PATTERN_NUM_SLICES, MAX_SLICE_BITS)
    
    # Write to file
    write_output_file(
        OUTPUT_FILE,
        table_slice_index,
        table_godel_bitmask,
        table_edge_godel_bitmask,
        table_inverse_cell_lookup,
        is_edge_cell,
        edge_index_of_cell
    )
    
    print(f"\n✅ Matcher tables written to: {OUTPUT_FILE}\n")
    
    # Print statistics
    print_statistics(
        table_slice_index,
        table_godel_bitmask,
        table_edge_godel_bitmask,
        table_inverse_cell_lookup,
        is_edge_cell
    )
    
    print("\n✨ Done!")