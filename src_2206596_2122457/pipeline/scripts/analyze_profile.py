"""
Profile Analysis Script

This script analyzes Python profiling statistics and generates a formatted report.

Usage:
    1. Generate profiling stats:
       python -m cProfile -o pipeline/out/profile_stats...
    
    2. Run this script:
       python pipeline/scripts/analyze_profile.py

Output:
    - Console: Formatted table of top functions by total time
    - File: pipeline/out/profile_analysis.txt (same table saved to disk)
"""

from __future__ import annotations
import os
import pstats
from pathlib import Path
from prettytable import PrettyTable
from typing import Any, Dict, List, Tuple, TypedDict, cast

SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "pipeline" / "out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_STATS_FILE: Path = OUTPUT_DIR / "profile_stats"

if not PROFILE_STATS_FILE.exists():
    print(f"❌ Error: Profile stats file not found at: {PROFILE_STATS_FILE}")
    print(f"   Please run profiling first:")
    print(
        f"   python -m cProfile -o {PROFILE_STATS_FILE} main_hex.py -t local .\\greedy_player_hex.py .\\my_player.py"
    )
    raise SystemExit(1)


class Record(TypedDict):
    function: str
    calls: int
    total_time: float
    per_call: float
    cum_time: float
    cum_per_call: float


StatsKey = Tuple[str, int, str]
StatsValue = Tuple[int, int, float, float, Dict[Any, Any]]

p: pstats.Stats = pstats.Stats(str(PROFILE_STATS_FILE))
p.sort_stats("tottime")

SORT_KEY: str = "total_time"
TOP_N: int = 500
PROJECT_PATH: str = str(PROJECT_ROOT / "src").lower()

print(PROJECT_PATH)

records: List[Record] = []

stats_dict: Dict[StatsKey, StatsValue] = cast(
    Dict[StatsKey, StatsValue], p.stats  # type: ignore
)

for func, stat_tuple in stats_dict.items():
    if len(func) != 3:
        continue
    if len(stat_tuple) < 5:
        continue

    file_path: str = func[0]
    line_no: int = func[1]
    func_name: str = func[2]

    cc: int = stat_tuple[0]
    nc: int = stat_tuple[1]
    tt: float = stat_tuple[2]
    ct: float = stat_tuple[3]
    callers: Dict[Any, Any] = stat_tuple[4]

    abs_path: str = os.path.abspath(file_path).lower()

    if PROJECT_PATH in abs_path and cc > 0:
        records.append(
            Record(
                function=f"{func_name} ({os.path.basename(file_path)}:{line_no})",
                calls=cc,
                total_time=tt,
                per_call=(tt / cc) if cc else 0.0,
                cum_time=ct,
                cum_per_call=(ct / cc) if cc else 0.0,
            )
        )

records.sort(
    key=lambda r: r.get(SORT_KEY, 0.0),
    reverse=True,
)

table = PrettyTable()
table.field_names = [
    "Function",
    "Calls",
    "Total Time (s)",
    "Per Call (s)",
    "Cumulative (s)",
    "Cum/Call (s)",
]

for r in records[:TOP_N]:
    table.add_row(
        [
            r["function"],
            r["calls"],
            f"{r['total_time']:.9f}",
            f"{r['per_call']:.9f}",
            f"{r['cum_time']:.9f}",
            f"{r['cum_per_call']:.9f}",
        ]
    )

print("\n📊 Profile Analysis")
print(f"   Project Root: {PROJECT_ROOT}")
print(f"   Profile File: {PROFILE_STATS_FILE}")
print(f"   Total Functions Found: {len(records)}\n")
print(table.get_string(title=f"Top {TOP_N} Project Functions by '{SORT_KEY}'")) # pyright: ignore[reportUnknownMemberType]
print()

output_file: Path = OUTPUT_DIR / "profile_analysis.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Profile Analysis\n")
    f.write(f"Project Root: {PROJECT_ROOT}\n")
    f.write(f"Profile File: {PROFILE_STATS_FILE}\n")
    f.write(f"Total Functions Found: {len(records)}\n\n")
    f.write(table.get_string(title=f"Top {TOP_N} Project Functions by '{SORT_KEY}'"))  # pyright: ignore[reportUnknownMemberType]

print(f"✅ Analysis saved to: {output_file}")