import os
from pathlib import Path

from src import config


def create_symlink(src: str | Path, dst: str | Path) -> None:
    try:
        os.symlink(src=src, dst=dst)
    except FileExistsError:
        pass


def normalize_pedestrian_no_pedestrian_dataset() -> None:
    raw_dataset_path = (
        config.data_raw_path / config.DatasetType.PEDESTRIAN_NO_PEDESTRIAN
    )
    config.data_normalized_path.mkdir(exist_ok=True)
    config.data_normalized_train_path.mkdir(exist_ok=True)
    config.data_normalized_test_path.mkdir(exist_ok=True)
    for file_path in os.scandir(
        raw_dataset_path / "data" / "train" / "no pedestrian"
    ):
        file_number = file_path.name.removeprefix("train (").removesuffix(
            ").jpg"
        )
        create_symlink(
            file_path.path,
            config.data_normalized_train_path / f"pnp-{file_number}-0.jpg",
        )

    for file_path in os.scandir(
        raw_dataset_path / "data" / "train" / "pedestrian"
    ):
        file_number = file_path.name.removeprefix("train (").removesuffix(
            ").jpg"
        )
        create_symlink(
            file_path.path,
            config.data_normalized_train_path / f"pnp-{file_number}-1.jpg",
        )

    for file_path in os.scandir(
        raw_dataset_path / "data" / "validation" / "no pedestrian"
    ):
        file_number = file_path.name.removeprefix("val (").removesuffix(").jpg")
        create_symlink(
            file_path.path,
            config.data_normalized_test_path / f"pnp-{file_number}-0.jpg",
        )

    for file_path in os.scandir(
        raw_dataset_path / "data" / "validation" / "pedestrian"
    ):
        file_number = file_path.name.removeprefix("val (").removesuffix(").jpg")
        create_symlink(
            file_path.path,
            config.data_normalized_test_path / f"pnp-{file_number}-1.jpg",
        )


def main() -> None:
    normalize_pedestrian_no_pedestrian_dataset()


if __name__ == "__main__":
    main()
