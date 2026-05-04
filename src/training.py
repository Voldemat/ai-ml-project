from typing import Any, cast
import numpy as np

from src.shared.loss import create_compute_hinge_loss_function
from src.shared.prepared_dataset import load_prepared_dataset
from src.shared.training import TrainingResult, train_weights
from src.svm import create_compute_new_weights_function


def train_model(
    max_dataset_size: int | None,
    random_generator: np.random.Generator,
    c_parameter: float,
    initial_learning_rate: float,
    decay_rate: float,
    epochs: int,
) -> TrainingResult:
    train_inputs, train_outputs = load_prepared_dataset(
        "train", max_dataset_size
    )
    compute_new_weights_function = create_compute_new_weights_function(
        c_parameter=c_parameter,
        initial_learning_rate=initial_learning_rate,
        decay_rate=decay_rate,
    )
    return train_weights(
        inputs=train_inputs,
        initial_weights=cast(
            np.typing.NDArray[np.floating[Any]],
            random_generator.normal(
                0, 0.01, size=cast(int, train_inputs.shape[1])
            ),
        ),
        outputs=train_outputs,
        epochs=epochs,
        compute_new_weights_function=compute_new_weights_function,
        compute_loss_function=create_compute_hinge_loss_function(c_parameter),
    )
