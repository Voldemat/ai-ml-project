import os

from typing import Literal

from pathlib import Path

import numpy as np


from src import config


def load_prepared_dataset(
    dataset_type: Literal["test", "train"],
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    inputs: list[np.typing.NDArray[np.float64]] = []
    outputs: list[np.typing.NDArray[np.float64]] = []
    for file_path in os.scandir(
        config.data_processed_train_path
        if dataset_type == "train"
        else config.data_processed_test_path
    ):
        inputs.append(np.load(file_path))
        outputs.append(
            np.array(
                [Path(file_path).stem.split("-")[-1] == "1"], dtype=np.float64
            )
        )

    return np.array(inputs), np.array(outputs)
