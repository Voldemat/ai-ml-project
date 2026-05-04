from typing import Any
import cv2
import numpy as np
from sklearn.svm import LinearSVC

from src.shared.hog.main import compute_hog
from src.shared.utils import prepare_image_for_hog
from src import config


def load_model() -> LinearSVC:
    weights: np.typing.NDArray[np.floating[Any]] = np.load(
        config.models_path / "weights.npy"
    )
    loaded_intercept = weights[:, 0]
    loaded_coef = weights[:, 1:]

    svc = LinearSVC()
    svc.coef_ = loaded_coef
    svc.intercept_ = loaded_intercept
    svc.classes_ = np.array([0.0, 1.0])
    return svc


def run(svc: LinearSVC, image: cv2.typing.MatLike) -> bool:
    prepared_image = prepare_image_for_hog(image)
    hog_array = compute_hog(prepared_image, cell_size=8, bin_count=9)
    return svc.predict(hog_array.reshape(1, -1))[0] == 1.0
