import os
import time
from typing import List


class IterationLogger:
    def __init__(self, base_dir: str, iteration_index: int) -> None:
        ts = int(time.time() * 1000)
        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, f"iter_{iteration_index:06d}_{ts}.txt")
        self._lines: List[str] = []

    def write(self, line: str) -> None:
        self._lines.append(line)

    def write_block(self, header: str, content: str) -> None:
        self._lines.append(header)
        self._lines.append(content)

    def flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines))

