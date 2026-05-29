import csv
from collections import Counter
from pathlib import Path

from config import (
    LABEL_MAP,
    METADATA_COLUMNS,
    METADATA_PATH,
    PROJECT_ROOT,
    RAVDESS_EMOTION_ID_MAP,
    RAW_DATA_DIR,
    SPLITS_DIR,
    TEST_ACTORS,
    TEST_SPLIT_PATH,
    TRAIN_ACTORS,
    TRAIN_SPLIT_PATH,
    VAL_ACTORS,
    VAL_SPLIT_PATH,
)


def resolve_raw_data_dir() -> Path:
    if RAW_DATA_DIR.exists():
        return RAW_DATA_DIR
    raise FileNotFoundError(f"RAVDESS directory not found: {RAW_DATA_DIR}")


def parse_ravdess_filename(path: Path) -> tuple[str, int]:
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename format: {path.name}")

    emotion_id = parts[2]
    actor = int(parts[6])
    return emotion_id, actor


def parse_folder_emotion(path: Path) -> str:
    folder = path.parent.name
    parts = folder.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid emotion folder format: {folder}")
    return parts[1]


def build_metadata(raw_data_dir: Path) -> list[dict[str, str | int]]:
    rows = []

    for wav_path in sorted(raw_data_dir.rglob("*.wav")):
        emotion_id, actor = parse_ravdess_filename(wav_path)
        emotion = RAVDESS_EMOTION_ID_MAP.get(emotion_id)

        if emotion is None or emotion == "calm":
            continue
        if emotion not in LABEL_MAP:
            continue

        folder_emotion = parse_folder_emotion(wav_path)
        if folder_emotion != emotion:
            raise ValueError(
                "Emotion mismatch: "
                f"folder={folder_emotion}, filename={emotion}, path={wav_path}"
            )

        rows.append(
            {
                "path": wav_path.relative_to(PROJECT_ROOT).as_posix(),
                "filename": wav_path.name,
                "emotion": emotion,
                "label": LABEL_MAP[emotion],
                "actor": actor,
            }
        )

    if not rows:
        raise ValueError(f"No valid RAVDESS wav files found in {raw_data_dir}")

    return rows


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def speaker_based_split(
    rows: list[dict[str, str | int]],
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]], list[dict[str, str | int]]]:
    train_actors = set(TRAIN_ACTORS)
    val_actors = set(VAL_ACTORS)
    test_actors = set(TEST_ACTORS)
    configured_actors = train_actors | val_actors | test_actors

    if len(configured_actors) != len(TRAIN_ACTORS) + len(VAL_ACTORS) + len(TEST_ACTORS):
        raise ValueError("Speaker split actor lists must not overlap")

    train_rows = []
    val_rows = []
    test_rows = []

    for row in rows:
        actor = int(row["actor"])
        if actor in train_actors:
            train_rows.append(row)
        elif actor in val_actors:
            val_rows.append(row)
        elif actor in test_actors:
            test_rows.append(row)
        else:
            raise ValueError(f"Actor {actor} is not assigned to any split")

    return train_rows, val_rows, test_rows


def validate_outputs(
    metadata_rows: list[dict[str, str | int]],
    train_rows: list[dict[str, str | int]],
    val_rows: list[dict[str, str | int]],
    test_rows: list[dict[str, str | int]],
) -> None:
    expected_emotions = set(LABEL_MAP)
    metadata_emotions = {str(row["emotion"]) for row in metadata_rows}
    if metadata_emotions != expected_emotions:
        raise ValueError(
            f"Unexpected emotions: expected={sorted(expected_emotions)}, "
            f"actual={sorted(metadata_emotions)}"
        )

    all_split_rows = train_rows + val_rows + test_rows
    if len(all_split_rows) != len(metadata_rows):
        raise ValueError("Split row count does not match metadata row count")

    paths = [str(row["path"]) for row in all_split_rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate paths found across train/val/test splits")

    for split_name, split_rows in (
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ):
        split_emotions = {str(row["emotion"]) for row in split_rows}
        if split_emotions != expected_emotions:
            raise ValueError(
                f"{split_name} split is missing classes: "
                f"{sorted(expected_emotions - split_emotions)}"
            )

    actor_sets = {
        "train": {int(row["actor"]) for row in train_rows},
        "val": {int(row["actor"]) for row in val_rows},
        "test": {int(row["actor"]) for row in test_rows},
    }
    expected_actor_sets = {
        "train": set(TRAIN_ACTORS),
        "val": set(VAL_ACTORS),
        "test": set(TEST_ACTORS),
    }

    for split_name, actors in actor_sets.items():
        if actors != expected_actor_sets[split_name]:
            raise ValueError(
                f"{split_name} actors mismatch: "
                f"expected={sorted(expected_actor_sets[split_name])}, "
                f"actual={sorted(actors)}"
            )

    if actor_sets["train"] & actor_sets["val"]:
        raise ValueError("Actor overlap found between train and val splits")
    if actor_sets["train"] & actor_sets["test"]:
        raise ValueError("Actor overlap found between train and test splits")
    if actor_sets["val"] & actor_sets["test"]:
        raise ValueError("Actor overlap found between val and test splits")


def print_distribution(name: str, rows: list[dict[str, str | int]]) -> None:
    counts = Counter(str(row["emotion"]) for row in rows)
    actors = sorted({int(row["actor"]) for row in rows})
    distribution = ", ".join(
        f"{emotion}={counts[emotion]}" for emotion in sorted(LABEL_MAP)
    )
    print(f"{name}: total={len(rows)} | actors={actors} | {distribution}")


def main() -> None:
    raw_data_dir = resolve_raw_data_dir()
    metadata_rows = build_metadata(raw_data_dir)
    train_rows, val_rows, test_rows = speaker_based_split(metadata_rows)

    validate_outputs(metadata_rows, train_rows, val_rows, test_rows)

    write_csv(METADATA_PATH, metadata_rows)
    write_csv(TRAIN_SPLIT_PATH, train_rows)
    write_csv(VAL_SPLIT_PATH, val_rows)
    write_csv(TEST_SPLIT_PATH, test_rows)

    print(f"Raw data directory: {raw_data_dir.relative_to(PROJECT_ROOT)}")
    print(f"Saved metadata: {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved splits: {SPLITS_DIR.relative_to(PROJECT_ROOT)}")
    print_distribution("metadata", metadata_rows)
    print_distribution("train", train_rows)
    print_distribution("val", val_rows)
    print_distribution("test", test_rows)


if __name__ == "__main__":
    main()
