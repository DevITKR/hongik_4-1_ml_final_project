# SER Project

Speech Emotion Recognition project using the RAVDESS dataset and a planned
Wav2Vec2 + BiLSTM + Multi-Head Attention model.

## Quick Run

Run the full preprocessing, training, and final evaluation flow:

```bash
python3 src/preprocess.py
PYTHONPATH=src python3 src/train.py
PYTHONPATH=src python3 src/evaluate.py
PYTHONPATH=src uvicorn api:app --host 0.0.0.0 --port 8000
python3 -m http.server 5173 --directory frontend
```

What each command does:

- `python3 src/preprocess.py`: creates metadata and speaker-based train/validation/test split CSV files.
- `PYTHONPATH=src python3 src/train.py`: trains with train/validation splits and saves the best validation checkpoint.
- `PYTHONPATH=src python3 src/evaluate.py`: evaluates the best checkpoint on the test split and saves metrics, report, and confusion matrix.
- `PYTHONPATH=src uvicorn api:app --host 0.0.0.0 --port 8000`: starts the API server for frontend audio emotion classification.
- `python3 -m http.server 5173 --directory frontend`: serves the simple HTML frontend at `http://localhost:5173`.

Smoke test commands:

```bash
PYTHONPATH=src python3 src/train.py --epochs 1 --batch-size 1 --max-train-batches 1 --max-val-batches 1 --device cpu
PYTHONPATH=src python3 src/evaluate.py --batch-size 1 --max-test-batches 1 --device cpu
```

Main generated files:

```text
data/processed/metadata.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
saved_models/wav2vec2_bilstm_attention_best.pth
results/metrics.json
results/classification_report.txt
results/confusion_matrix.png
```

`src/train.py` does not use the test split. The test split is used only by
`src/evaluate.py` after training is complete.

## Workflow

Implemented:

1. RAVDESS dataset scan
2. Audio metadata preprocessing
3. Speaker-based train / validation / test split
4. Wav2Vec2 weighted-layer feature extraction scaffold
5. Wav2Vec2 + BiLSTM + Multi-Head Attention model
6. Training with validation-based best model saving
7. Final test evaluation
8. Lightweight API server for frontend audio emotion classification
9. Simple HTML frontend for emotion acting feedback

## Dataset Layout

The target project layout is:

```text
ser_project/
  data/
    raw/
      RAVDESS/
    processed/
      metadata.csv
    splits/
      train.csv
      val.csv
      test.csv
```

The local dataset is stored at:

```text
data/raw/RAVDESS/
```

`src/preprocess.py` expects the RAVDESS dataset at `data/raw/RAVDESS/`.

The `calm` class is excluded. The active emotion classes are:

| Emotion | Label |
| --- | ---: |
| neutral | 0 |
| happy | 1 |
| sad | 2 |
| angry | 3 |
| fearful | 4 |
| disgust | 5 |
| surprised | 6 |

## Preprocessing

Run:

```bash
python3 src/preprocess.py
```

The script scans all `.wav` files under the resolved RAVDESS directory and
creates:

```text
data/processed/metadata.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

### Metadata Columns

The metadata is intentionally minimal:

| Column | Meaning |
| --- | --- |
| `path` | Project-root-relative path to the wav file |
| `filename` | Original wav filename |
| `emotion` | Emotion class name |
| `label` | Integer label used for model training |
| `actor` | RAVDESS actor id parsed from the filename |

Example:

```csv
path,filename,emotion,label,actor
data/raw/RAVDESS/01-neutral/03-01-01-01-01-01-01.wav,03-01-01-01-01-01-01.wav,neutral,0,1
```

The script parses the RAVDESS filename format:

```text
03-01-05-01-01-01-07.wav
```

Only `emotion_id` and `actor` are stored in the final metadata. The emotion id
is used to map the file to the final class label, and actor is kept for later
analysis or leakage checks.

The script also validates that the emotion parsed from the filename matches the
emotion folder name, for example:

```text
data/raw/RAVDESS/05-angry/03-01-05-01-01-01-07.wav
```

## Speaker-Based Train / Validation / Test Split

The split is actor-based to reduce speaker leakage between train, validation,
and test sets.

Configuration:

```text
train actors: 1-18
validation actors: 19-21
test actors: 22-24
```

Current generated split:

| Split | Samples |
| --- | ---: |
| metadata | 1248 |
| train | 936 |
| validation | 156 |
| test | 156 |

Class distribution after preprocessing:

| Emotion | Total | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| angry | 192 | 144 | 24 | 24 |
| disgust | 192 | 144 | 24 | 24 |
| fearful | 192 | 144 | 24 | 24 |
| happy | 192 | 144 | 24 | 24 |
| neutral | 96 | 72 | 12 | 12 |
| sad | 192 | 144 | 24 | 24 |
| surprised | 192 | 144 | 24 | 24 |

The script validates:

- all expected classes are present
- `calm` is excluded
- train / validation / test row counts match metadata
- no duplicate file path exists across splits
- every split contains all 7 emotion classes
- actor ids do not overlap between train, validation, and test

Validation and test usage policy:

- validation set: used every epoch for best model selection and hyperparameter tuning
- test set: used only once after all training and tuning are complete

## Notes

Preprocessing does not modify, copy, resample, normalize, pad, or truncate audio
files. Audio loading and conversion to the Wav2Vec2 input format is handled in
`src/dataset.py`.

## Wav2Vec2 Feature Extraction

Implemented files:

```text
src/dataset.py
src/model.py
requirements.txt
```

### Dependencies

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The current implementation uses:

- `torch`
- `torchaudio`
- `transformers`
- `numpy`
- `tqdm`
- `soundfile`

`soundfile` is used as a fallback audio loader because some recent `torchaudio`
versions require `torchcodec` for `torchaudio.load()`.

### Dataset

`RAVDESSDataset` reads one of the split CSV files:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

For each sample, it:

1. loads the wav file from the `path` column
2. converts multi-channel audio to mono
3. resamples audio to 16 kHz
4. crops or pads the waveform to 4 seconds
5. returns `(waveform, label)`

Current audio settings:

```text
sample rate: 16000
max audio seconds: 4
max audio samples: 64000
```

Smoke test result:

```text
train 936 (64000,) 0
val 156 (64000,) 0
test 156 (64000,) 0
```

### Wav2Vec2 Backbone

The model scaffold uses:

```text
facebook/wav2vec2-base
hidden size: 768
freeze: true
weighted layer sum: true
hidden states: 13
```

`src/model.py` provides `Wav2Vec2FeatureExtractor`, which takes a waveform batch
with shape:

```text
[batch, samples]
```

and returns Wav2Vec2 sequence features:

```text
[batch, time_steps, 768]
```

The feature extractor calls Wav2Vec2 with `output_hidden_states=True`. It uses
all 13 hidden states from `facebook/wav2vec2-base`: the initial projected feature
state plus 12 transformer layer outputs. These 13 tensors are combined with a
trainable softmax-normalized scalar weight vector. Wav2Vec2 itself remains
frozen, while the layer weights are trainable.

Smoke test result with batch size 2:

```text
waveforms (2, 64000)
features (2, 199, 768)
layer_weights (13,) True
all_wav2vec2_params_frozen True
```

### Freeze vs Fine-tune

This project currently uses Wav2Vec2 as a frozen feature extractor.

Freeze means:

- Wav2Vec2 extracts speech features but its weights are not updated
- only the downstream BiLSTM, Multi-Head Attention, and classifier will be trained
- training is faster and uses less memory
- overfitting risk is lower on a small dataset like RAVDESS
- implementation and debugging are simpler

Fine-tune means:

- Wav2Vec2 weights are updated together with the classifier
- the model can adapt more directly to emotion classification
- potential final performance can be higher
- training is slower and uses more GPU memory
- overfitting risk is higher on RAVDESS
- learning rate and regularization become more sensitive

Initial strategy:

```text
Use frozen Wav2Vec2 first.
Learn a weighted sum over 13 Wav2Vec2 hidden states.
Train BiLSTM + Multi-Head Attention + classifier on top of the weighted features.
Consider partial fine-tuning later only after the baseline is working.
```

Weighted layer sum is used because different Wav2Vec2 layers can encode
different levels of speech information. Instead of manually choosing one layer,
the model learns which combination of frozen layers is most useful for emotion
classification.

## Wav2Vec2 + BiLSTM + Multi-Head Attention Model

`src/model.py` now provides `SERModel`, the end-to-end emotion classifier.

Model flow:

```text
waveform [B, 64000]
-> frozen Wav2Vec2 13-layer weighted sum [B, T, 768]
-> 2-layer bidirectional LSTM [B, T, 512]
-> multi-head self-attention [B, T, 512]
-> residual connection + layer normalization [B, T, 512]
-> temporal mean pooling [B, 512]
-> FC classifier [B, 7]
```

Current model settings:

```text
num classes: 7
LSTM hidden size: 256
LSTM layers: 2
LSTM dropout: 0.3
attention heads: 4
attention dropout: 0.3
classifier hidden size: 256
classifier dropout: 0.3
```

`SERModel.forward(waveforms)` returns logits with shape:

```text
[batch, 7]
```

`SERModel.forward(waveforms, return_attention=True)` returns:

```text
(logits, attention_weights)
```

The Wav2Vec2 backbone remains frozen. The Wav2Vec2 layer weights, BiLSTM,
attention layer, and classifier are trainable.

## Training

`src/train.py` trains `SERModel` using only the train split and validates every
epoch using only the validation split. The test split is not loaded in
`train.py`.

Run full training:

```bash
PYTHONPATH=src python3 src/train.py
```

Useful smoke test command:

```bash
PYTHONPATH=src python3 src/train.py --epochs 1 --batch-size 1 --max-train-batches 1 --max-val-batches 1 --device cpu
```

Training configuration:

```text
optimizer: AdamW
scheduler: CosineAnnealingLR
loss: CrossEntropyLoss with train-split class weights
gradient clipping max norm: 1.0
default epochs: 30
default batch size: 8
learning rate: 1e-4
weight decay: 1e-2
```

Best model checkpoint:

```text
saved_models/wav2vec2_bilstm_attention_best.pth
```

Checkpoint contents:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
best_val_acc
config
label_map
```

Validation and test usage policy:

- validation set: used every epoch for best model selection
- test set: reserved for final evaluation only

## Final Test Evaluation

`src/evaluate.py` loads the best checkpoint and evaluates it on the test split.
This is the only workflow stage that uses `data/splits/test.csv`.

Run final evaluation:

```bash
PYTHONPATH=src python3 src/evaluate.py
```

Useful smoke test command:

```bash
PYTHONPATH=src python3 src/evaluate.py --batch-size 1 --max-test-batches 1 --device cpu
```

Evaluation outputs:

```text
results/metrics.json
results/classification_report.txt
results/confusion_matrix.png
```

Metrics saved in `metrics.json`:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_precision
weighted_recall
weighted_f1
test_sample_count
checkpoint_path
best_val_acc
confusion_matrix
labels
```

The classification report includes per-class precision, recall, F1-score, and
support. The confusion matrix uses this fixed label order:

```text
neutral, happy, sad, angry, fearful, disgust, surprised
```

If evaluating a checkpoint produced by a smoke test, the metric values are only
useful for verifying the evaluation pipeline. Final reported performance should
be produced from a checkpoint created by full training.

## API Server

`src/api.py` provides a lightweight FastAPI server for frontend audio emotion
classification. The API loads this checkpoint:

```text
saved_models/wav2vec2_bilstm_attention_best.pth
```

Run the server:

```bash
PYTHONPATH=src uvicorn api:app --host 0.0.0.0 --port 8000
```

For local-only testing:

```bash
PYTHONPATH=src uvicorn api:app --host 127.0.0.1 --port 8000
```

The server uses the same audio input policy as training and evaluation:

```text
mono conversion
16 kHz resampling
4 second crop/pad
model input shape: [1, 64000]
```

The API needs a trained best checkpoint. If the checkpoint does not exist,
`/health` still responds, but `/predict` returns `503 Service Unavailable`.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "mps"
}
```

Fields:

| Field | Meaning |
| --- | --- |
| `status` | API server process status |
| `model_loaded` | Whether the checkpoint was loaded successfully |
| `device` | Inference device selected by the server |

### Emotion Prediction

```http
POST /predict
Content-Type: multipart/form-data
```

Request parameters:

| Name | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `file` | `File` | yes | none | Audio file to classify. `.wav` is recommended. Browser-recorded `audio/webm` may fail if the local audio backend cannot decode it. |
| `return_probabilities` | `boolean` | no | `true` | If `true`, returns probabilities for all 7 emotion classes. |
| `return_top_k` | `integer` | no | `3` | Number of top predictions to return. Must be between `1` and `7`. |

Frontend request example:

```ts
const formData = new FormData();
formData.append("file", audioBlob, "recording.wav");
formData.append("return_probabilities", "true");
formData.append("return_top_k", "3");

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

const result = await response.json();
```

Do not send JSON for `/predict`. The frontend must send
`multipart/form-data`, and the browser will set the `Content-Type` boundary
automatically when `FormData` is used.

Example response:

```json
{
  "emotion": "angry",
  "label": 3,
  "confidence": 0.8123,
  "probabilities": {
    "neutral": 0.0123,
    "happy": 0.0312,
    "sad": 0.0441,
    "angry": 0.8123,
    "fearful": 0.0602,
    "disgust": 0.0217,
    "surprised": 0.0182
  },
  "top_k": [
    {
      "emotion": "angry",
      "label": 3,
      "probability": 0.8123
    },
    {
      "emotion": "fearful",
      "label": 4,
      "probability": 0.0602
    },
    {
      "emotion": "sad",
      "label": 2,
      "probability": 0.0441
    }
  ]
}
```

Response fields:

| Field | Meaning |
| --- | --- |
| `emotion` | Predicted emotion class name |
| `label` | Predicted integer label |
| `confidence` | Probability of the predicted label |
| `probabilities` | Class probability map, included only when `return_probabilities=true` |
| `top_k` | Top-k predictions sorted by probability |

Label mapping:

| Emotion | Label |
| --- | ---: |
| neutral | 0 |
| happy | 1 |
| sad | 2 |
| angry | 3 |
| fearful | 4 |
| disgust | 5 |
| surprised | 6 |

Error responses:

| Status | Cause |
| ---: | --- |
| `400` | Uploaded audio could not be decoded, or the uploaded file is empty |
| `422` | Required file is missing, or `return_top_k` is outside `1..7` |
| `503` | Best checkpoint is missing or failed to load |
| `500` | Model inference failed unexpectedly |

Smoke test:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/raw/RAVDESS/01-neutral/03-01-01-01-01-01-01.wav" \
  -F "return_probabilities=true" \
  -F "return_top_k=3"
```

## HTML Frontend

`frontend/index.html` is a static browser UI for emotion acting feedback. It
can record microphone audio with `MediaRecorder` or upload a local audio file,
then sends the audio to the API server.

Start the API server first:

```bash
PYTHONPATH=src uvicorn api:app --host 0.0.0.0 --port 8000
```

Start the frontend server:

```bash
python3 -m http.server 5173 --directory frontend
```

Open:

```text
http://localhost:5173
```

The frontend sends this request:

```http
POST http://localhost:8000/predict
Content-Type: multipart/form-data
```

Exact `FormData` parameters:

| Name | Value sent by frontend | Required | Description |
| --- | --- | --- | --- |
| `file` | recorded `Blob` or uploaded `File` | yes | Audio input selected by the user |
| `return_probabilities` | `"true"` | no | Requests all 7 class probabilities |
| `return_top_k` | `"3"` | no | Requests the top 3 predicted emotions |

Frontend request code:

```js
const formData = new FormData();
formData.append("file", audioBlobOrFile, "recording.wav");
formData.append("return_probabilities", "true");
formData.append("return_top_k", "3");

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});
```

The UI consumes these response fields:

| Field | Used for |
| --- | --- |
| `emotion` | Main predicted emotion and feedback selection |
| `label` | Integer class id display/debugging |
| `confidence` | Main confidence percentage |
| `probabilities` | Seven emotion probability bars |
| `top_k` | Top-3 candidate list |

Browser recording may produce `audio/webm`. If the backend audio decoder cannot
decode that format, use the upload control with a `.wav` file for stable tests.
