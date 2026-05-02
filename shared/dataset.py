import os
from pathlib import Path
from collections.abc import Iterator

dataset_path = Path(__file__).parent / '..' / 'dataset'

def get_train_dataset_directory() -> Path:
    return dataset_path / 'Train' / 'JPEGImages'

def get_train_dataset_files_paths() -> Iterator[os.DirEntry[str]]:
    return os.scandir(get_train_dataset_directory())

def get_test_dataset_directory() -> Path:
    return dataset_path / 'Test' / 'JPEGImages'

def get_test_dataset_files_paths() -> Iterator[os.DirEntry[str]]:
    return os.scandir(get_test_dataset_directory())

