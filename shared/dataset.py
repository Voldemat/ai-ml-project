from dataclasses import dataclass
import os
from pathlib import Path
from collections.abc import Iterator
import xml.etree.ElementTree as ET

dataset_path = Path(__file__).parent / ".." / "dataset"


def get_train_dataset_directory() -> Path:
    return dataset_path / "Train" / "JPEGImages"


def get_train_dataset_files_paths() -> Iterator[os.DirEntry[str]]:
    return os.scandir(get_train_dataset_directory())


def get_test_dataset_directory() -> Path:
    return dataset_path / "Test" / "JPEGImages"


def get_test_dataset_files_paths() -> Iterator[os.DirEntry[str]]:
    return os.scandir(get_test_dataset_directory())


@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class Metadata:
    bounding_box: list[BoundingBox]


def load_metadata_for_filepath(filepath: str) -> Metadata:
    f = Path(filepath)
    annotations_filepath = (
        Path(filepath).parent / ".." / "Annotations" / (f.stem + ".xml")
    )
    tree = ET.parse(annotations_filepath)
    root = tree.getroot()
    bounding_box: list[BoundingBox] = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        assert bndbox != 0
    return Metadata(bounding_box=bounding_box)
