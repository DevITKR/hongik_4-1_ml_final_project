import argparse
from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    GRAD_CLIP_MAX_NORM,
    LABEL_MAP,
    LEARNING_RATE,
    NUM_EPOCHS,
    RANDOM_SEED,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
    WEIGHT_DECAY,
)
from dataset import RAVDESSDataset
from model import SERModel
from utils import (
    accuracy_from_logits,
    compute_class_weights,
    get_device,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SERModel on RAVDESS splits.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_MAX_NORM)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def make_dataloader(
    dataset: RAVDESSDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def iter_limited(
    dataloader: DataLoader, max_batches: int | None
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


def train_one_epoch(
    model: SERModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        iter_limited(dataloader, max_batches),
        desc="train",
        leave=False,
        total=max_batches or len(dataloader),
    )
    for waveforms, labels in progress:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(waveforms)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite train loss encountered: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        correct, batch_total = accuracy_from_logits(logits.detach(), labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_total
        progress.set_postfix(
            loss=total_loss / max(total_samples, 1),
            acc=total_correct / max(total_samples, 1),
        )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def validate(
    model: SERModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        iter_limited(dataloader, max_batches),
        desc="val",
        leave=False,
        total=max_batches or len(dataloader),
    )
    for waveforms, labels in progress:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        logits = model(waveforms)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite validation loss encountered: {loss.item()}")

        correct, batch_total = accuracy_from_logits(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_total
        progress.set_postfix(
            loss=total_loss / max(total_samples, 1),
            acc=total_correct / max(total_samples, 1),
        )

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    train_dataset = RAVDESSDataset(TRAIN_SPLIT_PATH)
    val_dataset = RAVDESSDataset(VAL_SPLIT_PATH)
    train_loader = make_dataloader(train_dataset, args.batch_size, True, device)
    val_loader = make_dataloader(val_dataset, args.batch_size, False, device)

    class_weights = compute_class_weights(train_dataset.rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = SERModel().to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_val_acc = -1.0
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_max_norm": args.grad_clip,
        "num_classes": len(LABEL_MAP),
        "train_split_path": str(TRAIN_SPLIT_PATH),
        "val_split_path": str(VAL_SPLIT_PATH),
    }

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"class weights: {[round(float(weight), 4) for weight in class_weights.cpu()]}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
            args.max_train_batches,
        )
        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
            args.max_val_batches,
        )
        scheduler.step()

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                training_config,
            )

        marker = "*" if improved else " "
        print(
            f"{marker} epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"best_val_acc={best_val_acc:.4f}"
        )

    print(f"best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
