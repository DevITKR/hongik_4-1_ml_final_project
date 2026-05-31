from io import BytesIO
from typing import Any

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import BEST_MODEL_PATH, LABEL_MAP, MAX_AUDIO_SAMPLES, SAMPLE_RATE
from model import SERModel
from utils import get_device


ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app = FastAPI(title="SER Emotion Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = get_device("auto")
LABEL_NAMES = [label for label, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
MODEL: SERModel | None = None
MODEL_LOAD_ERROR: str | None = None


def load_model() -> SERModel:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {BEST_MODEL_PATH}. "
            "Run `PYTHONPATH=src python3 src/train.py` first."
        )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = SERModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@app.on_event("startup")
def startup() -> None:
    global MODEL, MODEL_LOAD_ERROR

    try:
        MODEL = load_model()
        MODEL_LOAD_ERROR = None
    except Exception as exc:
        MODEL = None
        MODEL_LOAD_ERROR = str(exc)


def decode_audio(file_bytes: bytes) -> tuple[torch.Tensor, int]:
    try:
        audio, sample_rate = sf.read(
            BytesIO(file_bytes),
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(audio).transpose(0, 1)
        return waveform, sample_rate
    except Exception as exc:
        soundfile_error = exc

    try:
        return torchaudio.load(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode audio file. Use a wav file if possible; "
                "browser-recorded webm may not be supported by the local audio backend."
            ),
        ) from soundfile_error or exc


def preprocess_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if waveform.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Expected audio shape [channels, samples], got {tuple(waveform.shape)}.",
        )

    waveform = waveform.mean(dim=0)
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=SAMPLE_RATE,
        )
        waveform = resampler(waveform)

    num_samples = waveform.shape[-1]
    if num_samples > MAX_AUDIO_SAMPLES:
        waveform = waveform[:MAX_AUDIO_SAMPLES]
    elif num_samples < MAX_AUDIO_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, MAX_AUDIO_SAMPLES - num_samples))

    return waveform.unsqueeze(0)


def get_loaded_model() -> SERModel:
    if MODEL is None:
        detail = MODEL_LOAD_ERROR or "Model is not loaded."
        raise HTTPException(status_code=503, detail=detail)
    return MODEL


def normalize_top_k(return_top_k: int) -> int:
    if return_top_k < 1 or return_top_k > len(LABEL_NAMES):
        raise HTTPException(
            status_code=422,
            detail=f"return_top_k must be between 1 and {len(LABEL_NAMES)}.",
        )
    return return_top_k


def build_prediction_response(
    probabilities: torch.Tensor,
    return_probabilities: bool,
    return_top_k: int,
) -> dict[str, Any]:
    probability_values = probabilities.cpu().tolist()
    predicted_label = int(torch.argmax(probabilities).item())
    confidence = float(probability_values[predicted_label])
    top_values, top_indices = torch.topk(probabilities, k=return_top_k)

    response: dict[str, Any] = {
        "emotion": LABEL_NAMES[predicted_label],
        "label": predicted_label,
        "confidence": confidence,
        "top_k": [
            {
                "emotion": LABEL_NAMES[int(label)],
                "label": int(label),
                "probability": float(probability),
            }
            for probability, label in zip(top_values.cpu().tolist(), top_indices.cpu().tolist())
        ],
    }

    if return_probabilities:
        response["probabilities"] = {
            label_name: float(probability_values[label_id])
            for label_id, label_name in enumerate(LABEL_NAMES)
        }

    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_probabilities: bool = Form(True),
    return_top_k: int = Form(3),
) -> dict[str, Any]:
    model = get_loaded_model()
    top_k = normalize_top_k(return_top_k)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    waveform, sample_rate = decode_audio(file_bytes)
    input_values = preprocess_waveform(waveform, sample_rate).to(DEVICE)

    try:
        with torch.no_grad():
            logits = model(input_values)
            probabilities = torch.softmax(logits.squeeze(0), dim=0)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    return build_prediction_response(probabilities, return_probabilities, top_k)
