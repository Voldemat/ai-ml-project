import sys

import cv2
import numpy as np
from sklearn.svm import LinearSVC

from src.shared.hog.main import compute_hog
from src.shared.utils import prepare_image_for_hog
from src import config


def main() -> None:
    image = cv2.imread(sys.argv[1])
    assert image is not None
    prepared_image = prepare_image_for_hog(image)
    hog_array = compute_hog(prepared_image, cell_size=8, bin_count=9)
    weights = np.load(config.models_path / "weights.npy")
    loaded_intercept = weights[:, 0]
    loaded_coef = weights[:, 1:]

    svc = LinearSVC()
    svc.coef_ = loaded_coef
    svc.intercept_ = loaded_intercept
    svc.classes_ = np.array([0.0, 1.0])

    has_pedestrian = svc.predict(hog_array.reshape(1, -1))[0] == 1.0
    print(f"Has pedestrian: {has_pedestrian}")


if __name__ == "__main__":
    main()
