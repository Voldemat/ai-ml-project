from typing import Any
import numpy as np


def accuracy_score(
    test_dataset: np.typing.NDArray[Any],
    predicted_dataset: np.typing.NDArray[Any],
) -> np.typing.NDArray[Any]:
    return np.mean(test_dataset == predicted_dataset)


def classification_report(
    test_dataset: np.typing.NDArray[Any],
    predicted_dataset: np.typing.NDArray[Any],
):
    """Generates a text report showing main classification metrics."""
    labels = np.unique(np.concatenate((test_dataset, predicted_dataset)))

    report = f"{'class':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"
    report += "-" * 55 + "\n"

    precisions, recalls, f1s, supports = [], [], [], []

    for label in labels:
        # True Positives, False Positives, False Negatives
        tp = np.sum((test_dataset == label) & (predicted_dataset == label))
        fp = np.sum((test_dataset != label) & (predicted_dataset == label))
        fn = np.sum((test_dataset == label) & (predicted_dataset != label))
        support = np.sum(test_dataset == label)

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1 = 2 * (P * R) / (P + R)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        report += f"{str(label):>10} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}\n"

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    # Add Macro Averages
    report += "-" * 55 + "\n"
    report += f"{'accuracy':>10} {'':>10} {'':>10} {np.mean(test_dataset == predicted_dataset):>10.2f} {len(test_dataset):>10}\n"
    report += f"{'macro avg':>10} {np.mean(precisions):>10.2f} {np.mean(recalls):>10.2f} {np.mean(f1s):>10.2f} {len(test_dataset):>10}\n"

    return report
