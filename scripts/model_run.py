import sys
from typing import Any

import cv2

import numpy as np

from src import config, inference


def main() -> None:
    image = cv2.imread(sys.argv[1])
    assert image is not None
    weights: np.typing.NDArray[Any] = np.load(
        config.models_path / "weights.npy"
    )
    has_pedestrian = inference.run(weights, image)
    print(f"Has pedestrian: {has_pedestrian}")


if __name__ == "__main__":
    main()
