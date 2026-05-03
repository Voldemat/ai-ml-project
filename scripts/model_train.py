from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import numpy as np

from src.shared.prepared_dataset import load_prepared_dataset
from src import config


def main() -> None:
    config.models_path.mkdir(exist_ok=True)
    train_inputs, train_outputs = load_prepared_dataset("train")
    test_inputs, test_outputs = load_prepared_dataset("test")
    svc = LinearSVC(C=0.1, random_state=42)
    svc.fit(train_inputs, train_outputs)
    predicted_outputs = svc.predict(test_inputs)
    weights = np.hstack((svc.intercept_.reshape(-1, 1), svc.coef_))
    np.save(config.models_path / "weights.npy", weights)

    accuracy = accuracy_score(test_outputs, predicted_outputs)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(test_outputs, predicted_outputs))


if __name__ == "__main__":
    main()
