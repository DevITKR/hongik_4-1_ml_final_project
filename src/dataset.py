import csv
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

try:
    from config import MAX_AUDIO_SAMPLES, PROJECT_ROOT, SAMPLE_RATE
except ModuleNotFoundError:
    from src.config import MAX_AUDIO_SAMPLES, PROJECT_ROOT, SAMPLE_RATE


class RAVDESSDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        sample_rate: int = SAMPLE_RATE,
        max_audio_samples: int = MAX_AUDIO_SAMPLES,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.sample_rate = sample_rate
        self.max_audio_samples = max_audio_samples
        self.rows = self._load_rows(self.csv_path)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        waveform, original_sample_rate = self._load_audio(PROJECT_ROOT / row["path"])
        waveform = self._to_mono(waveform)
        waveform = self._resample_if_needed(waveform, original_sample_rate)
        waveform = self._crop_or_pad(waveform)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return waveform, label

    @staticmethod
    def _load_rows(csv_path: Path) -> list[dict[str, str]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        with csv_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        if not rows:
            raise ValueError(f"Split CSV is empty: {csv_path}")

        required_columns = {"path", "label"}
        missing_columns = required_columns - set(rows[0])
        if missing_columns:
            raise ValueError(
                f"Split CSV is missing columns {sorted(missing_columns)}: {csv_path}"
            )

        return rows

    @staticmethod
    def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
        try:
            return torchaudio.load(path)
        except Exception:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio).transpose(0, 1)
            return waveform, sample_rate

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [channels, samples], got {waveform.shape}")
        return waveform.mean(dim=0)

    def _resample_if_needed(
        self, waveform: torch.Tensor, original_sample_rate: int
    ) -> torch.Tensor:
        if original_sample_rate == self.sample_rate:
            return waveform

        if original_sample_rate not in self._resamplers:
            self._resamplers[original_sample_rate] = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=self.sample_rate,
            )

        return self._resamplers[original_sample_rate](waveform)

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[-1]

        if num_samples > self.max_audio_samples:
            return waveform[: self.max_audio_samples]

        if num_samples < self.max_audio_samples:
            padding = self.max_audio_samples - num_samples
            return torch.nn.functional.pad(waveform, (0, padding))

        return waveform
