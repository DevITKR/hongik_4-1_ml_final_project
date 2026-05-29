import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    LABEL_MAP,
    METRICS_PATH,
    TEST_SPLIT_PATH,
)
from dataset import RAVDESSDataset
from model import SERModel
from utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SERModel on test split.")
    parser.add_argument("--checkpoint", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def get_label_names() -> list[str]:
    return [label for label, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run `PYTHONPATH=src python3 src/train.py` first."
        )
    return torch.load(path, map_location=device, weights_only=False)


def iter_limited(dataloader: DataLoader, max_batches: int | None):
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


@torch.no_grad()
def collect_predictions(
    model: SERModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[list[int], list[int], list[list[float]]]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []

    progress = tqdm(
        iter_limited(dataloader, max_batches),
        desc="test",
        leave=False,
        total=max_batches or len(dataloader),
    )
    for waveforms, labels in progress:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        logits = model(waveforms)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    return all_labels, all_predictions, all_probabilities


def save_confusion_matrix_plot(
    matrix: list[list[int]], label_names: list[str], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=range(len(label_names)),
        yticks=range(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    max_value = max(max(row) for row in matrix) if matrix else 0
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_metrics(
    labels: list[int],
    predictions: list[int],
    label_names: list[str],
    checkpoint: dict,
    checkpoint_path: Path,
) -> None:
    label_ids = list(range(len(label_names)))
    accuracy = accuracy_score(labels, predictions)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=label_ids,
            average="weighted",
            zero_division=0,
        )
    )
    matrix = confusion_matrix(labels, predictions, labels=label_ids).tolist()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "test_sample_count": len(labels),
        "checkpoint_path": str(checkpoint_path),
        "best_val_acc": checkpoint.get("best_val_acc"),
        "confusion_matrix": matrix,
        "labels": label_names,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
    )
    with CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    save_confusion_matrix_plot(matrix, label_names, CONFUSION_MATRIX_PATH)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    dataset = RAVDESSDataset(TEST_SPLIT_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = SERModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    labels, predictions, _ = collect_predictions(
        model,
        dataloader,
        device,
        args.max_test_batches,
    )
    label_names = get_label_names()
    save_metrics(labels, predictions, label_names, checkpoint, args.checkpoint)

    print(f"device: {device}")
    print(f"test samples evaluated: {len(labels)}")
    print(f"metrics saved to: {METRICS_PATH}")
    print(f"classification report saved to: {CLASSIFICATION_REPORT_PATH}")
    print(f"confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
