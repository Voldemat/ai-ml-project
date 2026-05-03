from typing import Any

import cv2

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


def prepare_image_for_hog(
    image: cv2.typing.MatLike,
) -> np.typing.NDArray[np.float32]:
    return cv2.cvtColor(
        cv2.cvtColor(cv2.resize(image, (64, 128)), cv2.COLOR_BGR2RGB),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)
