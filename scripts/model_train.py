import numpy as np

from src.shared.prepared_dataset import load_prepared_dataset
from src import config
from src.training import train_model
from src.shared.metrics import accuracy_score, classification_report


def main() -> None:
    config.models_path.mkdir(exist_ok=True)
    max_dataset_size: int | None = None
    training_result = train_model(
        max_dataset_size=max_dataset_size,
        random_generator=np.random.default_rng(42),
        c_parameter=1000,
        initial_learning_rate=0.0001,
        decay_rate=0.01,
        epochs=1000,
    )
    np.save(config.models_path / "weights.npy", training_result.weights)
    test_inputs, test_outputs = load_prepared_dataset("test", max_dataset_size)
    predicted_outputs = np.where(
        (test_inputs @ training_result.weights >= 0).astype(int) == 0,
        -1,
        1,
    ).reshape(-1, 1)
    print(
        f"Accuracy: {accuracy_score(test_outputs, predicted_outputs) * 100:.2f}%"
    )
    print(classification_report(test_outputs, predicted_outputs))


if __name__ == "__main__":
    main()
