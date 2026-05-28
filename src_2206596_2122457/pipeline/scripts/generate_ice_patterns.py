# ============================================================================
# REFACTORED: generate_ice_patterns.py
# ============================================================================

import os
from pathlib import Path
from typing import TextIO, TypedDict
import numpy as np
from src.ice_system.ice_constants import (
    PATTERN_NUM_SLICES, SLICE_NUM_FEATURE_TYPES, PATTERN_MAX_RADIUS,
    SLICE_FEATURE_CELLS, SLICE_FEATURE_RED_STONES, SLICE_FEATURE_BLUE_STONES,
    SLICE_FEATURE_PRIMARY_MARKED, SLICE_FEATURE_SECONDARY_MARKED,
    FLAG_HAS_EMPTY_CELLS, FLAG_HAS_PRIMARY_MOVES, FLAG_HAS_SECONDARY_MOVES,
    FLAG_HAS_WEIGHT_VALUE, CELL_EMPTY,
    TYPE_NEUTRAL_FILL, TYPE_BLUE_CAPTURE_FILL, TYPE_BLUE_STRONG_REVERSAL,
    TYPE_BLUE_THREAT_REVERSAL, TYPE_BLUE_INFERIOR_MOVE, TYPE_BLUE_REVERSIBLE,
    TYPE_VC_CAPTURE
)
from src.precomputed.ring_constants import VALID_GODELS

INPUT_FILE: str = "pipeline/data/ice-patterns.txt"
OUTPUT_DIR: str = "pipeline/out"
OUTPUT_FILE: str = os.path.join(OUTPUT_DIR, "precomputed_ice_patterns.py")


def compute_extension_from_godel(godel: int) -> int:
    for r in range(1, PATTERN_MAX_RADIUS + 1):
        max_bits = r * (r + 1) // 2
        if godel < (1 << max_bits):
            return r
    raise ValueError(f"Godel value {godel} exceeds max radius capacity")


def bit_positions_from_mask(mask: int) -> list[int]:
    positions: list[int] = []
    bit = 0
    while mask > 0:
        if mask & 1:
            positions.append(bit)
        mask >>= 1
        bit += 1
    return positions


def mask_subset(mask_small: int, mask_large: int) -> bool:
    return (mask_small & ~mask_large) == 0


def check_slice_valid(slice_bits: list[int]) -> bool:
    cells = int(slice_bits[SLICE_FEATURE_CELLS])
    red = int(slice_bits[SLICE_FEATURE_RED_STONES])
    blue = int(slice_bits[SLICE_FEATURE_BLUE_STONES])
    primary = int(slice_bits[SLICE_FEATURE_PRIMARY_MARKED])
    secondary = int(slice_bits[SLICE_FEATURE_SECONDARY_MARKED])
    
    if not mask_subset(red, cells) or not mask_subset(blue, cells):
        return False
    if (red & blue) != 0:
        return False
    if (primary & secondary) != 0:
        return False
    if not mask_subset(primary, cells) or not mask_subset(secondary, cells):
        return False
    
    return True


def compute_rotated_encodings(slices: np.ndarray) -> list[int]:
    encodings: list[int] = []
    for rotation in range(PATTERN_NUM_SLICES):
        encoding = 0
        for offset in range(PATTERN_NUM_SLICES):
            slice_idx = (rotation + offset) % PATTERN_NUM_SLICES
            slice_bits = int(slices[slice_idx, SLICE_FEATURE_CELLS])
            encoding |= (slice_bits & ((1 << 8) - 1)) << (offset * 8)
        encodings.append(encoding)
    return encodings


def compute_flags_for_pattern(slices: np.ndarray, weight_value: int) -> int:
    flags = 0
    has_empty = False
    has_primary = False
    has_secondary = False
    
    for s in range(PATTERN_NUM_SLICES):
        cells = int(slices[s, SLICE_FEATURE_CELLS])
        red = int(slices[s, SLICE_FEATURE_RED_STONES])
        blue = int(slices[s, SLICE_FEATURE_BLUE_STONES])
        primary = int(slices[s, SLICE_FEATURE_PRIMARY_MARKED])
        secondary = int(slices[s, SLICE_FEATURE_SECONDARY_MARKED])
        
        empty_mask = cells & ~(red | blue)
        if empty_mask != 0:
            has_empty = True
        if primary != 0:
            has_primary = True
        if secondary != 0:
            has_secondary = True
    
    if has_empty:
        flags |= FLAG_HAS_EMPTY_CELLS
    if has_primary:
        flags |= FLAG_HAS_PRIMARY_MOVES
    if has_secondary:
        flags |= FLAG_HAS_SECONDARY_MOVES
    if weight_value != 0:
        flags |= FLAG_HAS_WEIGHT_VALUE
    
    return flags


def flip_colors_slices(slices: np.ndarray) -> np.ndarray:
    """Flip red and blue colors in slices"""
    new_slices = slices.copy()
    for s in range(PATTERN_NUM_SLICES):
        new_slices[s, SLICE_FEATURE_RED_STONES], new_slices[s, SLICE_FEATURE_BLUE_STONES] = (
            new_slices[s, SLICE_FEATURE_BLUE_STONES],
            new_slices[s, SLICE_FEATURE_RED_STONES]
        )
    return new_slices


def compute_ring_godel(slices: np.ndarray, angle: int) -> int:
    """Compute gödel number for pattern at given rotation angle"""
    godel = 0
    offsets = np.arange(PATTERN_NUM_SLICES, dtype=np.int32)
    slice_indices = (angle + offsets) % PATTERN_NUM_SLICES
    cells_values = slices[slice_indices, SLICE_FEATURE_CELLS]
    
    for offset in range(PATTERN_NUM_SLICES):
        cells = int(cells_values[offset])
        shift = offset * 2  # BITS_PER_SLICE = 2
        godel |= (cells & 0x3) << shift
    
    return godel


def is_captured(slices: np.ndarray) -> bool:
    empty_cells = int(np.sum(slices[:, SLICE_FEATURE_CELLS] == CELL_EMPTY))
    primary_cells = int(np.sum(slices[:, SLICE_FEATURE_PRIMARY_MARKED] != 0))
    return empty_cells == primary_cells and empty_cells > 0


def is_vulnerable(slices: np.ndarray) -> bool:
    empty_cells = int(np.sum(slices[:, SLICE_FEATURE_CELLS] == CELL_EMPTY))
    return empty_cells == 1


class PatternData(TypedDict, total=False):
    type_code: str
    name: str
    comment: str
    weight_value: int
    slices: list[list[int]]


class ProcessedPattern(TypedDict):
    type_code: str
    name: str
    comment: str
    flags: int
    weight_value: int
    radius: int
    slices: np.ndarray
    rotated_encodings: list[int]


class IcePatternGenerator:
    
    def __init__(self) -> None:
        self.source_file = Path(INPUT_FILE)
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_source(self) -> list[PatternData]:
        patterns: list[PatternData] = []
        
        with open(self.source_file, 'r') as f:
            content = f.read()
        
        import re
        pattern_blocks = re.findall(r'\[([^\]]+)\]\s*([a-z]+):(.*?)(?=\[|$)', content, re.DOTALL)
        
        for name, type_code, data_str in pattern_blocks:
            name = name.strip()
            type_code = type_code.strip()
            data_str = data_str.strip()
            
            if not name or not type_code or not data_str:
                continue
            
            rows = [row.strip() for row in data_str.split(';') if row.strip()]
            slices: list[list[int]] = []
            for row in rows:
                row_values = [int(v) for v in row.split(',')]
                slices.append(row_values)
            
            if len(slices) == PATTERN_NUM_SLICES:
                pattern: PatternData = {
                    'type_code': type_code,
                    'name': name,
                    'comment': '',
                    'weight_value': 0,
                    'slices': slices
                }
                patterns.append(pattern)
        
        return patterns
    
    def _validate_and_process_pattern(self, pattern_data: PatternData) -> ProcessedPattern | None:
        if 'type_code' not in pattern_data or 'name' not in pattern_data:
            print(f"Warning: Pattern missing required fields: {pattern_data}")
            return None
        
        if 'slices' not in pattern_data or len(pattern_data['slices']) != PATTERN_NUM_SLICES:
            print(f"Warning: Pattern {pattern_data['name']} has invalid slices count")
            return None
        
        slices: list[list[int]] = []
        for s_idx, slice_data in enumerate(pattern_data['slices']):
            if len(slice_data) != SLICE_NUM_FEATURE_TYPES:
                print(f"Warning: Pattern {pattern_data['name']} slice {s_idx} has wrong feature count")
                return None
            if not check_slice_valid(slice_data):
                print(f"Warning: Pattern {pattern_data['name']} slice {s_idx} failed validation")
                return None
            slices.append(slice_data)
        
        slices_array = np.array(slices, dtype=np.uint32)
        
        godel_max = 0
        for s in range(PATTERN_NUM_SLICES):
            cells = int(slices_array[s, SLICE_FEATURE_CELLS])
            if cells > godel_max:
                godel_max = cells
        
        radius = compute_extension_from_godel(godel_max)
        weight_value = pattern_data.get('weight_value', 0)
        flags = compute_flags_for_pattern(slices_array, weight_value)
        rotated_encodings = compute_rotated_encodings(slices_array)
        
        return {
            'type_code': pattern_data['type_code'],
            'name': pattern_data['name'],
            'comment': pattern_data.get('comment', ''),
            'flags': flags,
            'weight_value': weight_value,
            'radius': radius,
            'slices': slices_array,
            'rotated_encodings': rotated_encodings
        }
    
    def _generate_pattern_variants(self, processed: list[ProcessedPattern]
    ) -> tuple[list[ProcessedPattern], dict[int, int]]:
        flipped_variants: list[ProcessedPattern] = []
        base_to_flipped_idx: dict[int, int] = {}
        
        for base_idx, pattern in enumerate(processed):
            flipped_slices = flip_colors_slices(pattern['slices'])
            flipped_flags = compute_flags_for_pattern(flipped_slices, pattern['weight_value'])
            flipped_rotated_encodings = compute_rotated_encodings(flipped_slices)
            
            flipped_variant: ProcessedPattern = {
                'type_code': pattern['type_code'],
                'name': pattern['name'] + '_flipped',
                'comment': pattern['comment'],
                'flags': flipped_flags,
                'weight_value': pattern['weight_value'],
                'radius': pattern['radius'],
                'slices': flipped_slices,
                'rotated_encodings': flipped_rotated_encodings
            }
            
            flipped_idx = len(processed) + len(flipped_variants)
            base_to_flipped_idx[base_idx] = flipped_idx
            flipped_variants.append(flipped_variant)
        
        return flipped_variants, base_to_flipped_idx
    
    def _categorize_patterns(self, processed: list[ProcessedPattern],
                            flipped_variants: list[ProcessedPattern]
    ) -> tuple[dict[str, list[int]], dict[int, int]]:
        categories: dict[str, list[int]] = {
            'e_fillin': [],
            'fillin_blue': [],
            'fillin_red': [],
            's_reversible_blue': [],
            's_reversible_red': [],
            't_reversible_blue': [],
            't_reversible_red': [],
            'inferior_blue': [],
            'inferior_red': [],
            'captured_blue': [],
            'captured_red': [],
            'vulnerable_blue': [],
            'vulnerable_red': [],
            'reversible_blue': [],
            'reversible_red': [],
        }
        
        pattern_to_categories: dict[int, int] = {}
        
        def add_pattern_to_categories(idx: int, pattern: ProcessedPattern, is_flipped: bool) -> None:
            type_code = pattern['type_code']
            slices = pattern['slices']
            flags = 0
            
            if type_code == TYPE_NEUTRAL_FILL:
                categories['e_fillin'].append(idx)
                flags |= 1
            
            elif type_code == TYPE_BLUE_CAPTURE_FILL:
                if not is_flipped:
                    categories['fillin_blue'].append(idx)
                    flags |= (1 << 1)
                    if is_captured(slices):
                        categories['captured_blue'].append(idx)
                        flags |= (1 << 9)
                else:
                    categories['fillin_red'].append(idx)
                    flags |= (1 << 2)
                    if is_captured(slices):
                        categories['captured_red'].append(idx)
                        flags |= (1 << 10)
            
            elif type_code == TYPE_BLUE_STRONG_REVERSAL:
                if not is_flipped:
                    categories['s_reversible_blue'].append(idx)
                    flags |= (1 << 3)
                    if is_vulnerable(slices):
                        categories['vulnerable_blue'].append(idx)
                        flags |= (1 << 11)
                else:
                    categories['s_reversible_red'].append(idx)
                    flags |= (1 << 4)
                    if is_vulnerable(slices):
                        categories['vulnerable_red'].append(idx)
                        flags |= (1 << 12)
            
            elif type_code == TYPE_BLUE_THREAT_REVERSAL:
                if not is_flipped:
                    categories['t_reversible_blue'].append(idx)
                    flags |= (1 << 5)
                    if is_vulnerable(slices):
                        categories['vulnerable_blue'].append(idx)
                        flags |= (1 << 11)
                else:
                    categories['t_reversible_red'].append(idx)
                    flags |= (1 << 6)
                    if is_vulnerable(slices):
                        categories['vulnerable_red'].append(idx)
                        flags |= (1 << 12)
            
            elif type_code == TYPE_BLUE_INFERIOR_MOVE:
                if not is_flipped:
                    categories['inferior_blue'].append(idx)
                    flags |= (1 << 7)
                else:
                    categories['inferior_red'].append(idx)
                    flags |= (1 << 8)
            
            elif type_code in (TYPE_BLUE_REVERSIBLE, TYPE_VC_CAPTURE):
                if not is_flipped:
                    categories['reversible_blue'].append(idx)
                    flags |= (1 << 13)
                else:
                    categories['reversible_red'].append(idx)
                    flags |= (1 << 14)
            
            pattern_to_categories[idx] = flags
        
        # Categorize base patterns
        for base_idx, pattern in enumerate(processed):
            add_pattern_to_categories(base_idx, pattern, False)
        
        # Categorize flipped variants
        for flipped_idx, pattern in enumerate(flipped_variants):
            add_pattern_to_categories(len(processed) + flipped_idx, pattern, True)
        
        return categories, pattern_to_categories
    
    def _build_category_godel_maps(self, category_patterns: dict[str, list[int]],
                                   all_patterns: list[ProcessedPattern],
                                   num_valid_godels: int) -> dict[str, list[list[tuple[int, int]]]]:
        godel_to_idx = {g: i for i, g in enumerate(VALID_GODELS)}
        category_godel_maps: dict[str, list[list[tuple[int, int]]]] = {}
        
        for cat_name, pattern_indices in category_patterns.items():
            godel_lookup: list[list[tuple[int, int]]] = [[] for _ in range(num_valid_godels)]
            
            for pattern_idx in pattern_indices:
                pattern = all_patterns[pattern_idx]
                slices = pattern['slices']
                
                for angle in range(PATTERN_NUM_SLICES):
                    godel = compute_ring_godel(slices, angle)
                    
                    if godel in godel_to_idx:
                        godel_idx = godel_to_idx[godel]
                        godel_lookup[godel_idx].append((pattern_idx, angle))
            
            category_godel_maps[cat_name] = godel_lookup
        
        return category_godel_maps
    
    def generate(self) -> None:
        source_patterns = self.parse_source()
        
        processed: list[ProcessedPattern] = []
        for pattern_data in source_patterns:
            result = self._validate_and_process_pattern(pattern_data)
            if result is not None:
                processed.append(result)
        
        if not processed:
            print("No patterns generated")
            return
        
        # Generate flipped variants
        flipped_variants, base_to_flipped = self._generate_pattern_variants(processed)
        
        # Combine all patterns
        all_patterns = processed + flipped_variants
        n_base = len(processed)
        n_total = len(all_patterns)
        
        # Categorize patterns
        category_patterns, pattern_categories = self._categorize_patterns(processed, flipped_variants)
        
        # Build gödel maps - pass combined patterns
        num_valid_godels = len(VALID_GODELS)
        category_godel_maps = self._build_category_godel_maps(
            category_patterns, all_patterns, num_valid_godels
        )
        
        # Prepare arrays for output
        names_arr = np.array([p['name'] for p in all_patterns], dtype=object)
        comments_arr = np.array([p['comment'] for p in all_patterns], dtype=object)
        type_codes_arr = np.array([p['type_code'] for p in all_patterns], dtype='U1')
        flags_arr = np.array([p['flags'] for p in all_patterns], dtype=np.uint32)
        weights_arr = np.array([p['weight_value'] for p in all_patterns], dtype=np.int32)
        radius_arr = np.array([p['radius'] for p in all_patterns], dtype=np.int8)
        slices_arr = np.stack([p['slices'] for p in all_patterns], axis=0)
        rotated_encodings_arr = np.array([p['rotated_encodings'] for p in all_patterns], dtype=np.uint64)
        
        # Create variant tracking arrays
        flipped_indices = np.full(n_total, -1, dtype=np.int32)
        for base_idx, flipped_idx in base_to_flipped.items():
            flipped_indices[base_idx] = flipped_idx
        
        pattern_category_flags = np.array(
            [pattern_categories.get(i, 0) for i in range(n_total)], dtype=np.uint32
        )
        
        output_file = Path(OUTPUT_FILE)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("import numpy as np\n\n")
            f.write("# ============================================================================\n")
            f.write("# PRECOMPUTED ICE PATTERNS - AUTO-GENERATED\n")
            f.write("# ============================================================================\n\n")
            
            self._write_array(f, 'PATTERNS_NAMES', names_arr, 'object')
            self._write_array(f, 'PATTERNS_COMMENTS', comments_arr, 'object')
            self._write_array(f, 'PATTERNS_TYPE_CODES', type_codes_arr, "'U1'")
            self._write_array(f, 'PATTERNS_FLAGS', flags_arr, 'np.uint32')
            self._write_array(f, 'PATTERNS_WEIGHTS', weights_arr, 'np.int32')
            self._write_array(f, 'PATTERNS_RADIUS', radius_arr, 'np.int8')
            self._write_array(f, 'PATTERN_CATEGORY_FLAGS', pattern_category_flags, 'np.uint32')
            
            f.write(f"PATTERNS_SLICES = np.array(\n")
            f.write(f"    {slices_arr.tolist()!r},\n")
            f.write(f"    dtype=np.uint32\n")
            f.write(f").reshape({n_total}, {PATTERN_NUM_SLICES}, {SLICE_NUM_FEATURE_TYPES})\n\n")
            
            f.write(f"PATTERNS_ROTATED_ENCODINGS = np.array(\n")
            f.write(f"    {rotated_encodings_arr.tolist()!r},\n")
            f.write(f"    dtype=np.uint64\n")
            f.write(f").reshape({n_total}, {PATTERN_NUM_SLICES})\n\n")
            
            self._write_array(f, 'PATTERNS_FLIPPED_INDEX', flipped_indices, 'np.int32')
            
            f.write("# Category assignments per pattern (bitflags)\n")
            f.write("# Bit 0: e_fillin\n")
            f.write("# Bit 1: fillin_blue, Bit 2: fillin_red\n")
            f.write("# Bit 3: s_reversible_blue, Bit 4: s_reversible_red\n")
            f.write("# Bit 5: t_reversible_blue, Bit 6: t_reversible_red\n")
            f.write("# Bit 7: inferior_blue, Bit 8: inferior_red\n")
            f.write("# Bit 9: captured_blue, Bit 10: captured_red\n")
            f.write("# Bit 11: vulnerable_blue, Bit 12: vulnerable_red\n")
            f.write("# Bit 13: reversible_blue, Bit 14: reversible_red\n\n")
            
            f.write("# Pattern indices per category\n\n")
            f.write("PATTERNS_E_FILLIN: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['e_fillin']))
            f.write("PATTERNS_FILLIN_BLUE: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['fillin_blue']))
            f.write("PATTERNS_FILLIN_RED: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['fillin_red']))
            f.write("PATTERNS_S_REVERSIBLE_BLUE: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['s_reversible_blue']))
            f.write("PATTERNS_S_REVERSIBLE_RED: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['s_reversible_red']))
            f.write("PATTERNS_T_REVERSIBLE_BLUE: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['t_reversible_blue']))
            f.write("PATTERNS_T_REVERSIBLE_RED: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['t_reversible_red']))
            f.write("PATTERNS_INFERIOR_BLUE: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['inferior_blue']))
            f.write("PATTERNS_INFERIOR_RED: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['inferior_red']))
            f.write("PATTERNS_CAPTURED_BLUE: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['captured_blue']))
            f.write("PATTERNS_CAPTURED_RED: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['captured_red']))
            f.write("PATTERNS_VULNERABLE_BLUE: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['vulnerable_blue']))
            f.write("PATTERNS_VULNERABLE_RED: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['vulnerable_red']))
            f.write("PATTERNS_REVERSIBLE_BLUE: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['reversible_blue']))
            f.write("PATTERNS_REVERSIBLE_RED: np.ndarray = np.array({}, dtype=np.int32)\n".format(category_patterns['reversible_red']))
            f.write("\n")
            
            f.write("# Precomputed gödel lookup tables per category\n")
            f.write("# [4096 valid_godels] → [(pattern_idx, rotation), ...]\n\n")
            for cat_name in sorted(category_godel_maps.keys()):
                var_name = f"GODEL_MAP_{cat_name.upper()}"
                godel_map = category_godel_maps[cat_name]
                f.write(f"{var_name}: list[list[tuple[int, int]]] = [\n")
                for _, patterns in enumerate(godel_map):
                    if patterns:
                        f.write(f"    {patterns!r},\n")
                    else:
                        f.write(f"    [],\n")
                f.write(f"]\n\n")
            
            f.write(f"NUM_BASE_PATTERNS = {n_base}\n")
            f.write(f"NUM_TOTAL_PATTERNS = {n_total}\n")
        
        print(f"Generated {n_total} patterns ({n_base} base + {len(flipped_variants)} flipped)")
        print(f"Output: {output_file}")
    
    def _write_array(self, f: TextIO, name: str, arr: np.ndarray, dtype_str: str) -> None:
        f.write(f"{name} = np.array(\n")
        f.write(f"    {arr.tolist()!r},\n")
        f.write(f"    dtype={dtype_str}\n")
        f.write(f")\n\n")


if __name__ == '__main__':
    generator = IcePatternGenerator()
    generator.generate()