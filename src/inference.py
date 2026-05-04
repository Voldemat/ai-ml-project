from typing import Any

import cv2

import numpy as np

from src.shared.bias import add_bias_term_to_input
from src.shared.hog.main import compute_hog
from src.shared.utils import prepare_image_for_hog


def run(weights: np.typing.NDArray[Any], image: cv2.typing.MatLike) -> bool:
    prepared_image = prepare_image_for_hog(image)
    hog_array = compute_hog(prepared_image, cell_size=8, bin_count=9)
    return bool((weights @ add_bias_term_to_input(hog_array)) >= 0)
