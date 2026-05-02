from typing import cast
import numpy as np

x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def compute_sobel(
    image: np.typing.NDArray[np.float32],
) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
    height, width = cast(tuple[int, int], image.shape)
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)

    for h in range(1, height - 1):
        for w in range(1, width - 1):
            region = image[h - 1 : h + 2, w - 1 : w + 2]
            gradient_x[h, w] = np.sum(region * x_kernel)
            gradient_y[h, w] = np.sum(region * y_kernel)

    return gradient_x, gradient_y
