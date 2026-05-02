from typing import Any
from collections.abc import Generator

import numpy as np

def normalize(
    cell_height: int,
    cell_width: int,
    cells: np.typing.NDArray[np.floating[Any]]
) -> Generator[np.typing.NDArray[np.float64], None, None]:
    block_size = 2
    for i in range(cell_height - block_size + 1):
        for j in range(cell_width - block_size + 1):
            block = cells[i:i+block_size, j:j+block_size].flatten()
            norm: np.floating = np.sqrt(np.sum(block**2) + 1e-6)
            yield block / norm
