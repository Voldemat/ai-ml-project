import os
from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import numpy as np

from shared.prepared_dataset import load_prepared_dataset


def main() -> None:
    train_inputs, train_outputs = load_prepared_dataset("train")
    test_inputs, test_outputs = load_prepared_dataset("test")
    svc = LinearSVC(C=0.01, random_state=42)
    svc.fit(train_inputs, train_outputs)
    predicted_outputs = svc.predict(test_inputs)
    np.save("weights.npy", svc.coef_)
    np.save("bias.npy", svc.intercept_)

    accuracy = accuracy_score(test_outputs, predicted_outputs)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(test_outputs, predicted_outputs))


if __name__ == "__main__":
    main()
