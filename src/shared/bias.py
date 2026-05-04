from typing import Any, cast

import numpy as np


def add_bias_term_to_inputs(
    inputs: np.typing.NDArray[np.floating[Any]],
) -> np.typing.NDArray[np.floating[Any]]:
    return cast(
        np.typing.NDArray[np.floating[Any]],
        np.c_[np.ones((inputs.shape[0], 1)), inputs],
    )
