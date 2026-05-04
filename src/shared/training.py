import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .loss import ComputeLossFunction, ComputeLossFunctionInputs


@dataclass
class ComputeNewWeightsFunctionInputs:
    inputs: np.typing.NDArray[np.floating[Any]]
    weights: np.typing.NDArray[np.floating[Any]]
    outputs: np.typing.NDArray[np.floating[Any]]
    epoch: int


ComputeNewWeightsFunction = Callable[
    [ComputeNewWeightsFunctionInputs],
    np.typing.NDArray[np.floating[Any]],
]


@dataclass
class TrainingResult:
    weights: np.typing.NDArray[np.floating[Any]]
    loss_history: list[float]
    training_time: float


def train_weights(
    inputs: np.typing.NDArray[np.floating[Any]],
    initial_weights: np.typing.NDArray[np.floating[Any]],
    outputs: np.typing.NDArray[np.floating[Any]],
    epochs: int,
    compute_new_weights_function: ComputeNewWeightsFunction,
    compute_loss_function: ComputeLossFunction,
) -> TrainingResult:
    assert epochs > 0
    start = time.perf_counter()
    loss_history: list[float] = []
    working_weights = initial_weights
    for epoch in tqdm(range(epochs)):
        working_weights = compute_new_weights_function(
            ComputeNewWeightsFunctionInputs(
                inputs=inputs,
                weights=working_weights,
                outputs=outputs,
                epoch=epoch,
            )
        )
        outputs_prediction = inputs @ working_weights
        loss = compute_loss_function(
            ComputeLossFunctionInputs(
                test_dataset=outputs,
                predicted_dataset=outputs_prediction,
                weights=working_weights,
            )
        )
        loss_history.append(loss)
    end = time.perf_counter()
    return TrainingResult(
        weights=working_weights,
        loss_history=loss_history,
        training_time=end - start,
    )
