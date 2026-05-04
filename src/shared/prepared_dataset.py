import os

from typing import Any, Literal

from pathlib import Path

import numpy as np


from src import config
from .bias import add_bias_term_to_inputs


def load_prepared_dataset(
    dataset_type: Literal["test", "train"],
    max_dataset_size: int | None
) -> tuple[
    np.typing.NDArray[np.floating[Any]], np.typing.NDArray[np.floating[Any]]
]:
    inputs: list[np.typing.NDArray[np.floating[Any]]] = []
    outputs: list[np.typing.NDArray[np.floating[Any]]] = []
    for index, file_path in enumerate(os.scandir(
        config.data_processed_train_path
        if dataset_type == "train"
        else config.data_processed_test_path
    )):
        if max_dataset_size is not None and index >= max_dataset_size - 1:
            break
        inputs.append(np.load(file_path))
        outputs.append(
            np.array(
                [1 if Path(file_path).stem.split("-")[-1] == "1" else -1],
                dtype=np.float64,
            )
        )

    return add_bias_term_to_inputs(np.array(inputs)), np.array(outputs)
