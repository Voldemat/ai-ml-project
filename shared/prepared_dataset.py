import os

from typing import Literal

import numpy as np


def load_prepared_dataset(
    dataset_type: Literal["test", "train"],
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    rows: list[np.typing.NDArray[np.float64]] = []
    for file_path in os.scandir(f"prepared_dataset/{dataset_type}"):
        rows.append(np.load(file_path))

    dataset = np.array(rows)
    inputs = dataset[:, :-1]
    outputs = dataset[:, -1]
    return inputs, outputs
