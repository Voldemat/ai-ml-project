import os
from pathlib import Path
from typing import Any, cast
from shared.dataset import (
    get_train_dataset_files_paths,
    get_test_dataset_files_paths,
    load_metadata_for_filepath,
)
from shared.utils import prepare_image_for_hog
from shared.hog import compute_hog

import cv2

import numpy as np


def process_image(
    file_path: Path, output_base_path: Path, has_pedestrian: bool
) -> None:
    image = cv2.imread(file_path)
    assert image is not None
    prepared_image = prepare_image_for_hog(image)
    hog_array = compute_hog(prepared_image, cell_size=8, bin_count=9)
    filename = file_path.stem
    final_array = np.concatenate([hog_array, [float(has_pedestrian)]])
    np.save(
        output_base_path / f"{filename}_{int(has_pedestrian)}",
        final_array,
    )


if __name__ == "__main__":
    Path("prepared_dataset").mkdir(exist_ok=True)
    Path("prepared_dataset/train").mkdir(exist_ok=True)
    Path("prepared_dataset/test").mkdir(exist_ok=True)
    for file_path in os.scandir(
        Path("datasets/pedestrian-no-pedestrian/data/train/no pedestrian")
    ):
        process_image(
            Path(file_path),
            output_base_path=Path("prepared_dataset/train"),
            has_pedestrian=False,
        )
    for file_path in os.scandir(
        Path("datasets/pedestrian-no-pedestrian/data/train/pedestrian")
    ):
        process_image(
            Path(file_path),
            output_base_path=Path("prepared_dataset/train"),
            has_pedestrian=True,
        )

    for file_path in os.scandir(
        Path("datasets/pedestrian-no-pedestrian/data/validation/no pedestrian")
    ):
        process_image(
            Path(file_path),
            output_base_path=Path("prepared_dataset/test"),
            has_pedestrian=False,
        )
    for file_path in os.scandir(
        Path("datasets/pedestrian-no-pedestrian/data/validation/pedestrian")
    ):
        process_image(
            Path(file_path),
            output_base_path=Path("prepared_dataset/test"),
            has_pedestrian=True,
        )
