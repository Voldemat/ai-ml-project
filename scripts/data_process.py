import os
from pathlib import Path

from src.shared.utils import prepare_image_for_hog
from src.shared.hog import compute_hog
from src import config

import cv2

from tqdm import tqdm

import numpy as np


def process_image(file_path: Path, output_base_path: Path) -> None:
    image = cv2.imread(file_path)
    assert image is not None
    prepared_image = prepare_image_for_hog(image)
    hog_array = compute_hog(prepared_image, cell_size=8, bin_count=9)
    filename = file_path.stem
    np.save(
        output_base_path / filename,
        hog_array,
    )


def process_images(images_dir_path: Path, output_base_path: Path) -> None:
    print(
        "Processing images images_dir_path: "
        + f"{os.path.relpath(images_dir_path.resolve())}, "
        + f"output_base_path: {os.path.relpath(output_base_path.resolve())}"
    )
    for file_path in tqdm(
        os.scandir(images_dir_path),
        total=sum(1 for _ in os.scandir(images_dir_path)),
    ):
        process_image(
            Path(file_path),
            output_base_path=output_base_path,
        )


if __name__ == "__main__":
    config.data_processed_path.mkdir(exist_ok=True)
    config.data_processed_train_path.mkdir(exist_ok=True)
    config.data_processed_test_path.mkdir(exist_ok=True)
    process_images(
        config.data_normalized_train_path,
        config.data_processed_train_path,
    )
    process_images(
        config.data_normalized_test_path,
        config.data_processed_test_path,
    )
