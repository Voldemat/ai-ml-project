import enum
from pathlib import Path


root_path = Path(__file__).parent / ".."

data_path = root_path / "data"


class DatasetType(enum.StrEnum):
    INRIA_PERSON = "inria-person"
    PEDESTRIAN_NO_PEDESTRIAN = "pedestrian-no-pedestrian"


data_raw_path = data_path / "raw"

data_processed_path = data_path / "processed"
data_processed_train_path = data_processed_path / "train"
data_processed_test_path = data_processed_path / "test"

data_normalized_path = data_path / "normalized"
data_normalized_train_path = data_normalized_path / "train"
data_normalized_test_path = data_normalized_path / "test"

models_path = root_path / "models"
