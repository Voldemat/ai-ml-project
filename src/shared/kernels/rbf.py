from typing import Any

import numpy as np


def compute_rbf_kernel(
    gamma_parameter: float,
    inputs: np.typing.NDArray[np.floating[Any]],
    inputs_2: np.typing.NDArray[np.floating[Any]],
) -> np.typing.NDArray[np.floating[Any]]:
    inputs_norm = np.sum(inputs**2, axis=1).reshape(-1, 1)
    outputs_norm = np.sum(inputs_2**2, axis=1).reshape(1, -1)

    distances = (
        inputs_norm + outputs_norm - 2 * np.dot(inputs, inputs_2.transpose())
    )
    print(f"{inputs_norm=} {outputs_norm=} {distances=}")
    return np.exp(-gamma_parameter * distances)
