import os
from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import numpy as np


if __name__ == "__main__":
    train_rows: list[np.typing.NDArray[np.float64]] = []
    for file_path in os.scandir("prepared_dataset/train"):
        train_rows.append(np.load(file_path))

    train_dataset = np.array(train_rows)
    train_inputs = train_dataset[:, :-1]
    train_outputs = train_dataset[:, -1]

    test_rows: list[np.typing.NDArray[np.float64]] = []
    for file_path in os.scandir("prepared_dataset/test"):
        test_rows.append(np.load(file_path))

    test_dataset = np.array(test_rows)
    test_inputs = test_dataset[:, :-1]
    test_outputs = test_dataset[:, -1]

    svc = LinearSVC(C=0.01, random_state=42)
    svc.fit(train_inputs, train_outputs)
    predicted_outputs = svc.predict(test_inputs)
    np.save("weights.npy", svc.coef_)
    np.save("bias.npy", svc.intercept_)


    accuracy = accuracy_score(test_outputs, predicted_outputs)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(test_outputs, predicted_outputs))
