import numpy as np
import os
from typing import List, Tuple, Optional, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from numpy.typing import NDArray

INPUT_FILE: str = "pipeline/data/vc-patterns.txt"
OUTPUT_DIR: str = "pipeline/out"
OUTPUT_FILE: str = os.path.join(OUTPUT_DIR, "precomputed_vc_patterns.py")
BOARD_SIZE: int = 14


class TemplatePattern(NamedTuple):
    name: str
    pattern_type: str
    height: int
    rows: List[str]


@dataclass
class BuilderPattern:
    black_mask: NDArray[np.bool_]
    empty_mask: NDArray[np.bool_]
    endpoint: int
    height: int


@dataclass
class CompletePattern:
    must_have: NDArray[np.bool_]
    not_opponent: NDArray[np.bool_]
    endpoints: Tuple[int, int]


# Direction constants: EAST=0, WEST=1, NORTH=2, SOUTH=3
EAST = 0
WEST = 1
NORTH = 2
SOUTH = 3

@lru_cache(maxsize=8)
def build_index_map_for_shift(direction: int, width: int, height: int) -> NDArray[np.int32]:
    n_cells = width * height
    index_map = np.arange(n_cells, dtype=np.int32)
    
    for idx in range(n_cells):
        row = idx // width
        col = idx % width
        
        if direction == EAST:
            new_col = col + 1
            if new_col >= width:
                index_map[idx] = -1
            else:
                index_map[idx] = row * width + new_col
        elif direction == WEST:
            new_col = col - 1
            if new_col < 0:
                index_map[idx] = -1
            else:
                index_map[idx] = row * width + new_col
        elif direction == NORTH:
            new_row = row + 1
            if new_row >= height:
                index_map[idx] = -1
            else:
                index_map[idx] = new_row * width + col
        elif direction == SOUTH:
            new_row = row - 1
            if new_row < 0:
                index_map[idx] = -1
            else:
                index_map[idx] = new_row * width + col
    
    return index_map


@lru_cache(maxsize=8)
def build_index_map_for_rotate(width: int, height: int) -> NDArray[np.int32]:
    n_cells = width * height
    index_map = np.full(n_cells, -1, dtype=np.int32)
    
    for idx in range(n_cells):
        row = idx // width
        col = idx % width
        new_row = col
        new_col = height - 1 - row
        if new_row < height and new_col < width:
            index_map[idx] = new_row * width + new_col
    
    return index_map


@lru_cache(maxsize=8)
def build_index_map_for_mirror(width: int, height: int) -> NDArray[np.int32]:
    n_cells = width * height
    index_map = np.full(n_cells, -1, dtype=np.int32)
    
    for idx in range(n_cells):
        row = idx // width
        col = idx % width
        new_col = width - 1 - col
        index_map[idx] = row * width + new_col
    
    return index_map


@lru_cache(maxsize=8)
def build_index_map_for_reverse(width: int, height: int) -> NDArray[np.int32]:
    n_cells = width * height
    index_map = np.full(n_cells, -1, dtype=np.int32)
    
    for idx in range(n_cells):
        row = idx // width
        col = idx % width
        new_x = (width - 1 - col) + (height - 1 - row)
        if new_x < width:
            index_map[idx] = new_x * width + row
    
    return index_map


def coords_to_index(col: int, row: int, width: int) -> int:
    return row * width + col


def index_to_coords(index: int, width: int) -> Tuple[int, int]:
    row = index // width
    col = index % width
    return col, row


def apply_index_map_to_mask(mask: NDArray[np.bool_], index_map: NDArray[np.int32]) -> Optional[NDArray[np.bool_]]:
    result = np.zeros_like(mask)
    for old_idx in np.where(mask)[0]:
        new_idx = index_map[old_idx]
        if new_idx >= 0:
            result[new_idx] = True
        else:
            return None
    return result


def parse_pattern_templates(input_file: str) -> List[TemplatePattern]:
    templates: List[TemplatePattern] = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')
        
        if not line.strip() or line.strip().startswith('#'):
            i += 1
            continue
        
        if line.strip().startswith('name'):
            parts = line.strip().split()
            name = parts[1] if len(parts) > 1 else "unknown"
            i += 1
            
            pattern_type = ""
            height = 0
            
            while i < len(lines):
                attr_line = lines[i].rstrip('\n').strip()
                if attr_line.startswith('type'):
                    pattern_type = attr_line.split()[1]
                    i += 1
                elif attr_line.startswith('height'):
                    height_str = attr_line.split()[1]
                    height = int(height_str)
                    i += 1
                elif attr_line:
                    break
                else:
                    i += 1
                    break
            
            rows: List[str] = []
            
            if height == -1:
                while i < len(lines):
                    row_line = lines[i].rstrip('\n')
                    if not row_line.strip():
                        break
                    rows.append(row_line)
                    i += 1
                height = len(rows)
            else:
                for _ in range(height):
                    if i < len(lines):
                        row_line = lines[i].rstrip('\n')
                        rows.append(row_line)
                        i += 1
            
            if name and pattern_type and height > 0 and len(rows) > 0:
                templates.append(TemplatePattern(name, pattern_type, height, rows))
                print(f"Parsed: {name} type={pattern_type} height={height} rows={len(rows)}")
        else:
            i += 1
    
    return templates


def parse_template_to_masks(
    template: TemplatePattern, width: int, height: int
) -> Optional[Tuple[NDArray[np.bool_], NDArray[np.bool_], int]]:
    black_mask = np.zeros(width * height, dtype=np.bool_)
    empty_mask = np.zeros(width * height, dtype=np.bool_)
    endpoint = -1
    
    if len(template.rows) == 0:
        return None
    
    # Check if pattern is too tall
    if len(template.rows) > height:
        return None
    
    # Start at bottom of board and iterate backwards through rows (C++ style)
    row = height - 1
    for row_idx in range(len(template.rows) - 1, -1, -1):
        row_str = template.rows[row_idx]
        
        # Parse whitespace-delimited tokens (C++ style: "is >> sym")
        tokens = row_str.split()
        col = 0
        
        for token in tokens:
            if col >= width:
                return None
            
            if row < 0 or row >= height:
                return None
            
            abs_idx = coords_to_index(col, row, width)
            
            # Process each character in token (should be single char, but handle just in case)
            char = token[0]
            if char == 'B':
                black_mask[abs_idx] = True
            elif char == '*':
                empty_mask[abs_idx] = True
            elif char == 'E':
                endpoint = abs_idx
                empty_mask[abs_idx] = True
            elif char == '.':
                pass
            else:
                return None
            
            col += 1
        
        row -= 1
    
    if endpoint < 0:
        return None
    
    return black_mask, empty_mask, endpoint


def hex_adjacent(index1: int, index2: int, width: int, height: int) -> bool:
    """Check if two hex cells are orthogonally adjacent (6 neighbors in hex)."""
    if index1 == index2:
        return False
    
    col1, row1 = index_to_coords(index1, width)
    col2, row2 = index_to_coords(index2, width)
    
    # In a rectangular hex grid, adjacent cells are at most 1 step away in col and row
    col_diff = abs(col1 - col2)
    row_diff = abs(row1 - row2)
    
    # Adjacent if: (same row, adjacent col) or (same col, adjacent row) or (diagonal in hex)
    if col_diff == 0 and row_diff == 1:
        return True
    if col_diff == 1 and row_diff == 0:
        return True
    if col_diff == 1 and row_diff == 1:
        return True
    
    return False


def combine_start_and_end(
    start_patterns: List[BuilderPattern],
    end_patterns: List[BuilderPattern],
    width: int,
    height: int
) -> List[CompletePattern]:
    complete: List[CompletePattern] = []
    
    for start in start_patterns:
        for end in end_patterns:
            if end.height < start.height:
                continue
            
            bp_black = end.black_mask.copy()
            bp_empty = end.empty_mask.copy()
            bp_endpoint = end.endpoint
            
            on_board = True
            
            # Shift end pattern until it no longer overlaps with start
            while on_board:
                overlap = (start.black_mask | start.empty_mask) & (bp_black | bp_empty)
                if not overlap.any():
                    break
                
                shift_map = build_index_map_for_shift(EAST, width, height)
                shifted_black = apply_index_map_to_mask(bp_black, shift_map)
                shifted_empty = apply_index_map_to_mask(bp_empty, shift_map)
                
                if shifted_black is None or shifted_empty is None:
                    on_board = False
                    break
                
                col_shift, row_shift = index_to_coords(bp_endpoint, width)
                col_shift += 1
                if col_shift >= width:
                    on_board = False
                    break
                
                bp_black = shifted_black
                bp_empty = shifted_empty
                bp_endpoint = coords_to_index(col_shift, row_shift, width)
            
            if not on_board:
                continue
            
            start_col = 0
            col = 0
            on_board = True
            
            # Generate patterns by repeatedly shifting end further east
            while on_board:
                combined_empty = start.empty_mask | bp_empty
                combined_black = start.black_mask | bp_black
                
                # Fill cells between start and end (C++ style: range i < col, not i <= col)
                for i in range(start_col, col):
                    for j in range(start.height):
                        p_idx = coords_to_index(i, height - 1 - j, width)
                        if 0 <= p_idx < combined_empty.size:
                            combined_empty[p_idx] = True
                
                ep0 = start.endpoint
                ep1 = bp_endpoint
                
                # Only add if endpoints are NOT adjacent (C++ style: !Adjacent)
                if not hex_adjacent(ep0, ep1, width, height):
                    complete.append(CompletePattern(
                        combined_black, combined_empty,
                        (ep0, ep1)
                    ))
                
                shift_map = build_index_map_for_shift(EAST, width, height)
                shifted_black = apply_index_map_to_mask(bp_black, shift_map)
                shifted_empty = apply_index_map_to_mask(bp_empty, shift_map)
                
                if shifted_black is None or shifted_empty is None:
                    on_board = False
                    break
                
                col_shift, row_shift = index_to_coords(bp_endpoint, width)
                col_shift += 1
                if col_shift >= width:
                    on_board = False
                    break
                
                bp_black = shifted_black
                bp_empty = shifted_empty
                bp_endpoint = coords_to_index(col_shift, row_shift, width)
                col += 1
    
    return complete


def shift_and_add(
    pattern: CompletePattern,
    direction: int,
    width: int,
    height: int
) -> List[CompletePattern]:
    variants: List[CompletePattern] = []
    
    current_must = pattern.must_have.copy()
    current_not_opp = pattern.not_opponent.copy()
    current_ep = pattern.endpoints
    
    variants.append(pattern)
    
    while True:
        index_map = build_index_map_for_shift(direction, width, height)
        shifted_must = apply_index_map_to_mask(current_must, index_map)
        shifted_not_opp = apply_index_map_to_mask(current_not_opp, index_map)
        
        if shifted_must is None or shifted_not_opp is None:
            break
        
        ep0_old, ep1_old = current_ep
        col0, row0 = index_to_coords(ep0_old, width)
        col1, row1 = index_to_coords(ep1_old, width)
        
        if direction == EAST:
            col0, col1 = col0 + 1, col1 + 1
        elif direction == WEST:
            col0, col1 = col0 - 1, col1 - 1
        elif direction == NORTH:
            row0, row1 = row0 + 1, row1 + 1
        elif direction == SOUTH:
            row0, row1 = row0 - 1, row1 - 1
        
        if (0 <= col0 < width and 0 <= row0 < height and
            0 <= col1 < width and 0 <= row1 < height):
            ep0_new = coords_to_index(col0, row0, width)
            ep1_new = coords_to_index(col1, row1, width)
            current_ep = (ep0_new, ep1_new)
            current_must = shifted_must
            current_not_opp = shifted_not_opp
            variants.append(CompletePattern(current_must, current_not_opp, current_ep))
        else:
            break
    
    return variants


def rotate_pattern(pattern: CompletePattern, width: int, height: int) -> Optional[CompletePattern]:
    index_map = build_index_map_for_rotate(width, height)
    rotated_must = apply_index_map_to_mask(pattern.must_have, index_map)
    rotated_not = apply_index_map_to_mask(pattern.not_opponent, index_map)
    
    if rotated_must is None or rotated_not is None:
        return None
    
    ep0_col, ep0_row = index_to_coords(pattern.endpoints[0], width)
    ep1_col, ep1_row = index_to_coords(pattern.endpoints[1], width)
    
    ep0_new_row = ep0_col
    ep0_new_col = height - 1 - ep0_row
    ep1_new_row = ep1_col
    ep1_new_col = height - 1 - ep1_row
    
    ep0_new = coords_to_index(ep0_new_col, ep0_new_row, width)
    ep1_new = coords_to_index(ep1_new_col, ep1_new_row, width)
    
    return CompletePattern(rotated_must, rotated_not, (ep0_new, ep1_new))


def mirror_pattern(pattern: CompletePattern, width: int, height: int) -> CompletePattern:
    index_map = build_index_map_for_mirror(width, height)
    mirrored_must = apply_index_map_to_mask(pattern.must_have, index_map)
    mirrored_not = apply_index_map_to_mask(pattern.not_opponent, index_map)
    
    if mirrored_must is None or mirrored_not is None:
        return pattern
    
    ep0_col, ep0_row = index_to_coords(pattern.endpoints[0], width)
    ep1_col, ep1_row = index_to_coords(pattern.endpoints[1], width)
    
    ep0_col = width - 1 - ep0_col
    ep1_col = width - 1 - ep1_col
    
    ep0_new = coords_to_index(ep0_col, ep0_row, width)
    ep1_new = coords_to_index(ep1_col, ep1_row, width)
    
    return CompletePattern(mirrored_must, mirrored_not, (ep0_new, ep1_new))


def reverse_pattern(pattern: CompletePattern, width: int, height: int) -> Optional[CompletePattern]:
    index_map = build_index_map_for_reverse(width, height)
    reversed_must = apply_index_map_to_mask(pattern.must_have, index_map)
    reversed_not = apply_index_map_to_mask(pattern.not_opponent, index_map)
    
    if reversed_must is None or reversed_not is None:
        return None
    
    ep0_col, ep0_row = index_to_coords(pattern.endpoints[0], width)
    ep1_col, ep1_row = index_to_coords(pattern.endpoints[1], width)
    
    new_x0 = (width - 1 - ep0_col) + (height - 1 - ep0_row)
    new_x1 = (width - 1 - ep1_col) + (height - 1 - ep1_row)
    
    if new_x0 >= width or new_x1 >= width:
        return None
    
    ep0_new = coords_to_index(new_x0, ep0_row, width)
    ep1_new = coords_to_index(new_x1, ep1_row, width)
    
    return CompletePattern(reversed_must, reversed_not, (ep0_new, ep1_new))


def rotate_and_shift(
    pattern: CompletePattern,
    dir1: int,
    dir2: int,
    width: int,
    height: int
) -> List[CompletePattern]:
    variants: List[CompletePattern] = []
    
    for shifted in shift_and_add(pattern, dir1, width, height):
        variants.append(shifted)
    
    rotated = rotate_pattern(pattern, width, height)
    if rotated:
        for shifted in shift_and_add(rotated, dir2, width, height):
            variants.append(shifted)
    
    return variants


def expand_pattern_variants(
    patterns: List[CompletePattern],
    width: int,
    height: int
) -> Tuple[List[CompletePattern], List[CompletePattern]]:
    black_variants: List[CompletePattern] = []
    white_variants: List[CompletePattern] = []
    black_seen: set[Tuple[bytes, bytes, Tuple[int, int]]] = set()
    white_seen: set[Tuple[bytes, bytes, Tuple[int, int]]] = set()
    
    def add_to_black(pat: CompletePattern) -> None:
        packed_must = np.packbits(pat.must_have).tobytes()
        packed_not = np.packbits(pat.not_opponent).tobytes()
        key = (packed_must, packed_not, pat.endpoints)
        if key not in black_seen:
            black_seen.add(key)
            black_variants.append(pat)
    
    def add_to_white(pat: CompletePattern) -> None:
        packed_must = np.packbits(pat.must_have).tobytes()
        packed_not = np.packbits(pat.not_opponent).tobytes()
        key = (packed_must, packed_not, pat.endpoints)
        if key not in white_seen:
            white_seen.add(key)
            white_variants.append(pat)
    
    for pattern in patterns:
        for shifted in rotate_and_shift(pattern, EAST, WEST, width, height):
            add_to_black(shifted)
        
        mirrored = mirror_pattern(pattern, width, height)
        for shifted in rotate_and_shift(mirrored, SOUTH, NORTH, width, height):
            add_to_white(shifted)
        
        reversed_pat = reverse_pattern(pattern, width, height)
        if reversed_pat:
            for shifted in rotate_and_shift(reversed_pat, WEST, EAST, width, height):
                add_to_black(shifted)
        
        if reversed_pat:
            reversed_mirrored = mirror_pattern(reversed_pat, width, height)
            for shifted in rotate_and_shift(reversed_mirrored, NORTH, SOUTH, width, height):
                add_to_white(shifted)
    
    return black_variants, white_variants


def write_precomputed_arrays(
    out_file: str,
    patterns_black: List[CompletePattern],
    patterns_white: List[CompletePattern],
    width: int,
    height: int
) -> None:
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write(f"BOARD_WIDTH = {width}\n")
        f.write(f"BOARD_HEIGHT = {height}\n\n")
        
        f.write(f"NUM_PATTERNS_BLACK = {len(patterns_black)}\n")
        f.write(f"NUM_PATTERNS_WHITE = {len(patterns_white)}\n\n")
        
        f.write("PATTERNS_MUST_HAVE_BLACK = np.array([\n")
        for pattern in patterns_black:
            f.write(f"    {pattern.must_have.astype(int).tolist()},\n")
        f.write("], dtype=bool)\n\n")
        
        f.write("PATTERNS_NOT_OPPONENT_BLACK = np.array([\n")
        for pattern in patterns_black:
            f.write(f"    {pattern.not_opponent.astype(int).tolist()},\n")
        f.write("], dtype=bool)\n\n")
        
        f.write("PATTERNS_ENDPOINTS_BLACK = np.array([\n")
        for pattern in patterns_black:
            ep = pattern.endpoints
            f.write(f"    [{int(ep[0])}, {int(ep[1])}],\n")
        f.write("], dtype=np.int32)\n\n")
        
        f.write("PATTERNS_MUST_HAVE_WHITE = np.array([\n")
        for pattern in patterns_white:
            f.write(f"    {pattern.must_have.astype(int).tolist()},\n")
        f.write("], dtype=bool)\n\n")
        
        f.write("PATTERNS_NOT_OPPONENT_WHITE = np.array([\n")
        for pattern in patterns_white:
            f.write(f"    {pattern.not_opponent.astype(int).tolist()},\n")
        f.write("], dtype=bool)\n\n")
        
        f.write("PATTERNS_ENDPOINTS_WHITE = np.array([\n")
        for pattern in patterns_white:
            ep = pattern.endpoints
            f.write(f"    [{int(ep[0])}, {int(ep[1])}],\n")
        f.write("], dtype=np.int32)\n")


def main() -> None:
    templates = parse_pattern_templates(INPUT_FILE)
    print(f"Total templates parsed: {len(templates)}\n")
    
    if len(templates) == 0:
        print("ERROR: No templates found in file!")
        return
    
    complete_patterns: List[CompletePattern] = []
    start_patterns: List[BuilderPattern] = []
    end_patterns: List[BuilderPattern] = []
    failed_count = 0
    
    for template in templates:
        result = parse_template_to_masks(template, BOARD_SIZE, BOARD_SIZE)
        if result is None:
            print(f"  Skipped (too large): {template.name}")
            failed_count += 1
            continue
        
        black_mask, empty_mask, endpoint = result
        builder = BuilderPattern(black_mask, empty_mask, endpoint, template.height)
        
        if template.pattern_type == "complete":
            complete_patterns.append(CompletePattern(black_mask, empty_mask, (endpoint, endpoint)))
            print(f"  Added complete: {template.name}")
        elif template.pattern_type == "start":
            start_patterns.append(builder)
            print(f"  Added start: {template.name}")
        elif template.pattern_type == "end":
            end_patterns.append(builder)
            print(f"  Added end: {template.name}")
    
    print(f"\nSkipped {failed_count} patterns (too large for {BOARD_SIZE}x{BOARD_SIZE} board)")
    print(f"Parsed {len(complete_patterns)} complete patterns")
    print(f"Parsed {len(start_patterns)} start patterns")
    print(f"Parsed {len(end_patterns)} end patterns\n")
    
    combined = combine_start_and_end(start_patterns, end_patterns, BOARD_SIZE, BOARD_SIZE)
    print(f"Constructed {len(combined)} ladder patterns\n")
    
    all_patterns = complete_patterns + combined
    print(f"Total patterns before expansion: {len(all_patterns)}\n")
    
    if len(all_patterns) == 0:
        print("ERROR: No patterns to expand!")
        return
    
    print("Calling expand_pattern_variants...")
    patterns_black, patterns_white = expand_pattern_variants(all_patterns, BOARD_SIZE, BOARD_SIZE)
    
    print(f"\nExpanded to {len(patterns_black)} BLACK patterns")
    print(f"Expanded to {len(patterns_white)} WHITE patterns")
    
    write_precomputed_arrays(OUTPUT_FILE, patterns_black, patterns_white, BOARD_SIZE, BOARD_SIZE)
    print(f"Generated patterns to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()