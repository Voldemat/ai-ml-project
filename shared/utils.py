from typing import Any

import numpy as np


def cart_to_polar(
    x: np.typing.NDArray[np.floating[Any]],
    y: np.typing.NDArray[np.floating[Any]],
    angle_in_degrees: bool = True,
) -> tuple[
    np.typing.NDArray[np.floating[Any]], np.typing.NDArray[np.floating[Any]]
]:
    magnitude = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)

    if angle_in_degrees:
        angle = np.mod(np.degrees(angle), 360)

    return magnitude, angle
