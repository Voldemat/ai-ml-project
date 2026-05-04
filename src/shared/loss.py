from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class ComputeLossFunctionInputs:
    test_dataset: np.typing.NDArray[np.floating[Any]]
    predicted_dataset: np.typing.NDArray[np.floating[Any]]
    weights: np.typing.NDArray[np.floating[Any]]


ComputeLossFunction = Callable[[ComputeLossFunctionInputs], float]


def compute_hinge_loss(
    inputs: ComputeLossFunctionInputs,
    c_parameter: float,
) -> float:
    hinge = np.maximum(0, 1 - inputs.test_dataset * inputs.predicted_dataset)
    return float(
        0.5 * np.dot(inputs.weights, inputs.weights)
        + c_parameter * np.mean(hinge)
    )


def create_compute_hinge_loss_function(
    c_parameter: float,
) -> ComputeLossFunction:
    def wrapper(inputs: ComputeLossFunctionInputs) -> float:
        return compute_hinge_loss(
            inputs=inputs,
            c_parameter=c_parameter,
        )

    return wrapper
