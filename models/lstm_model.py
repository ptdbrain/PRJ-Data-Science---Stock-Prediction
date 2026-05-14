"""
LSTM Model cho dự đoán xu hướng giá TCB (Classification).
══════════════════════════════════════════════════════════
Output: raw logit (BCEWithLogitsLoss được áp dụng trong BasePredictor.fit)
        → Sigmoid → P(tăng) ∈ [0, 1]
"""
import torch
import torch.nn as nn
from models.base_model import BasePredictor
from config.settings import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class LSTMPredictor(BasePredictor):

    def __init__(self, **kwargs):
        super().__init__(model_name="lstm", **kwargs)

    def build_model(self, input_size: int) -> nn.Module:
        """Trả về LSTM model với output là raw logit (1 neuron)."""
        class LSTMNet(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS,
                    batch_first=True,
                    dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
                )
                self.fc = nn.Sequential(
                    nn.Linear(HIDDEN_SIZE, 64),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(64, 1),
                    # Không thêm Sigmoid ở đây vì dùng BCEWithLogitsLoss
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                return self.fc(last_hidden).squeeze(-1)  # raw logit

        return LSTMNet(input_size)
