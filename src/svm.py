from typing import Any
import numpy as np

from .shared.training import (
    ComputeNewWeightsFunction,
    ComputeNewWeightsFunctionInputs,
)


def compute_new_weights(
    c_parameter: float,
    learning_rate: float,
    inputs: np.typing.NDArray[np.floating[Any]],
    weights: np.typing.NDArray[np.floating[Any]],
    outputs: np.typing.NDArray[np.floating[Any]],
) -> np.typing.NDArray[np.floating[Any]]:
    working_weights = weights
    x_i: np.floating[Any]
    for i, x_i in enumerate(inputs):
        margin_ok: bool = outputs[i] * np.dot(x_i, working_weights) >= 1
        grad_w = (
            working_weights
            if margin_ok
            else working_weights - c_parameter * outputs[i] * x_i
        )
        working_weights -= learning_rate * grad_w
    return working_weights


def create_compute_new_weights_function(
    c_parameter: float,
    initial_learning_rate: float,
    decay_rate: float,
) -> ComputeNewWeightsFunction:
    def wrapper(
        inputs: ComputeNewWeightsFunctionInputs,
    ) -> np.typing.NDArray[np.floating[Any]]:
        return compute_new_weights(
            c_parameter,
            initial_learning_rate / (1 + decay_rate * inputs.epoch),
            inputs.inputs,
            inputs.weights,
            inputs.outputs,
        )

    return wrapper
