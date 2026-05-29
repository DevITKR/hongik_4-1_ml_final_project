import torch
from torch import nn

try:
    from config import (
        ATTENTION_DROPOUT,
        ATTENTION_NUM_HEADS,
        CLASSIFIER_DROPOUT,
        CLASSIFIER_HIDDEN_SIZE,
        FREEZE_WAV2VEC2,
        LSTM_DROPOUT,
        LSTM_HIDDEN_SIZE,
        LSTM_NUM_LAYERS,
        NUM_CLASSES,
        USE_WEIGHTED_LAYER_SUM,
        WAV2VEC2_HIDDEN_SIZE,
        WAV2VEC2_MODEL_NAME,
        WAV2VEC2_NUM_HIDDEN_STATES,
    )
except ModuleNotFoundError:
    from src.config import (
        ATTENTION_DROPOUT,
        ATTENTION_NUM_HEADS,
        CLASSIFIER_DROPOUT,
        CLASSIFIER_HIDDEN_SIZE,
        FREEZE_WAV2VEC2,
        LSTM_DROPOUT,
        LSTM_HIDDEN_SIZE,
        LSTM_NUM_LAYERS,
        NUM_CLASSES,
        USE_WEIGHTED_LAYER_SUM,
        WAV2VEC2_HIDDEN_SIZE,
        WAV2VEC2_MODEL_NAME,
        WAV2VEC2_NUM_HIDDEN_STATES,
    )

try:
    from transformers import Wav2Vec2Model
except ModuleNotFoundError as exc:
    Wav2Vec2Model = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = WAV2VEC2_MODEL_NAME,
        freeze: bool = FREEZE_WAV2VEC2,
        use_weighted_layer_sum: bool = USE_WEIGHTED_LAYER_SUM,
        num_hidden_states: int = WAV2VEC2_NUM_HIDDEN_STATES,
    ) -> None:
        super().__init__()

        if Wav2Vec2Model is None:
            raise ModuleNotFoundError(
                "transformers is required for Wav2Vec2 feature extraction. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.freeze = freeze
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.num_hidden_states = num_hidden_states
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.layer_weights = nn.Parameter(torch.zeros(num_hidden_states))

        if self.freeze:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.wav2vec2.eval()
        for parameter in self.wav2vec2.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "Wav2Vec2FeatureExtractor":
        super().train(mode)
        if self.freeze:
            self.wav2vec2.eval()
        return self

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError(
                f"Expected waveform batch shape [batch, samples], got {waveforms.shape}"
            )

        if self.freeze:
            with torch.no_grad():
                outputs = self.wav2vec2(
                    input_values=waveforms,
                    output_hidden_states=self.use_weighted_layer_sum,
                )
        else:
            outputs = self.wav2vec2(
                input_values=waveforms,
                output_hidden_states=self.use_weighted_layer_sum,
            )

        if not self.use_weighted_layer_sum:
            return outputs.last_hidden_state

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("Wav2Vec2 did not return hidden states")
        if len(hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"Expected {self.num_hidden_states} hidden states, "
                f"got {len(hidden_states)}"
            )

        stacked_hidden_states = torch.stack(hidden_states, dim=0)
        normalized_weights = torch.softmax(self.layer_weights, dim=0)
        weighted_features = torch.einsum(
            "l,lbtc->btc", normalized_weights, stacked_hidden_states
        )
        return weighted_features


class SERModel(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        lstm_hidden_size: int = LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = LSTM_NUM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        attention_num_heads: int = ATTENTION_NUM_HEADS,
        attention_dropout: float = ATTENTION_DROPOUT,
        classifier_hidden_size: int = CLASSIFIER_HIDDEN_SIZE,
        classifier_dropout: float = CLASSIFIER_DROPOUT,
    ) -> None:
        super().__init__()

        self.feature_extractor = Wav2Vec2FeatureExtractor()
        self.lstm_output_size = lstm_hidden_size * 2

        self.bilstm = nn.LSTM(
            input_size=WAV2VEC2_HIDDEN_SIZE,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm_norm = nn.LayerNorm(self.lstm_output_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_output_size,
            num_heads=attention_num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.lstm_output_size)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.lstm_output_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_size, num_classes),
        )

    def forward(
        self, waveforms: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(waveforms)
        lstm_output, _ = self.bilstm(features)
        lstm_output = self.lstm_norm(lstm_output)

        attention_output, attention_weights = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            need_weights=return_attention,
        )
        attended_features = self.attention_norm(lstm_output + attention_output)
        pooled_features = attended_features.mean(dim=1)
        logits = self.classifier(pooled_features)

        if return_attention:
            return logits, attention_weights

        return logits
