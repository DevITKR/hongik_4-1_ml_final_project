import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from config import LABEL_MAP, RANDOM_SEED
except ModuleNotFoundError:
    from src.config import LABEL_MAP, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    predictions = torch.argmax(logits, dim=1)
    correct = int((predictions == labels).sum().item())
    total = int(labels.numel())
    return correct, total


def compute_class_weights(
    rows: list[dict[str, str]], num_classes: int = len(LABEL_MAP)
) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    total = sum(counts.values())

    weights = []
    for label in range(num_classes):
        count = counts[label]
        if count == 0:
            raise ValueError(f"Cannot compute class weight for missing label {label}")
        weights.append(total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_acc: float,
    config: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": config,
            "label_map": LABEL_MAP,
        },
        checkpoint_path,
    )
