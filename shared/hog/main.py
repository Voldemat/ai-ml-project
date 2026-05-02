from typing import Any, cast

import numpy as np

from .sobel import compute_sobel
from .l2_normalization import normalize
from ..utils import cart_to_polar


def compute_hog(
    image_grayscale: np.typing.NDArray[np.floating[Any]],
    cell_size: int,
    bin_count: int,
) -> np.typing.NDArray[np.floating[Any]]:
    gradient_x, gradient_y = compute_sobel(image_grayscale)
    mag, angle = cart_to_polar(gradient_x, gradient_y, angle_in_degrees=True)
    height, width = cast(tuple[int, int], image_grayscale.shape)
    cell_height = height // cell_size
    cell_width = width // cell_size
    hog_cells = np.zeros((cell_height, cell_width, bin_count))

    for h in range(cell_height):
        for w in range(cell_width):
            mag_patch = mag[
                h * cell_size : (h + 1) * cell_size,
                w * cell_size : (w + 1) * cell_size,
            ]
            angle_patch = angle[
                h * cell_size : (h + 1) * cell_size,
                w * cell_size : (w + 1) * cell_size,
            ]
            cell_histogram, _ = np.histogram(
                angle_patch, bins=bin_count, range=(0, 180), weights=mag_patch
            )
            hog_cells[h, w, :] = cell_histogram
    return np.array(list(normalize(cell_height, cell_width, hog_cells))).flatten()
