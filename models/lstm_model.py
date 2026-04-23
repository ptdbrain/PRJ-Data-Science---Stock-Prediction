"""
LSTM Model cho dự đoán giá TCB.
═══════════════════════════════
Phụ trách: Thành viên D
Branch: feature/model-lstm
"""
import torch
import torch.nn as nn
from models.base_model import BasePredictor
from config.settings import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class LSTMPredictor(BasePredictor):

    def __init__(self, **kwargs):
        super().__init__(model_name="lstm", **kwargs)

    def build_model(self, input_size: int) -> nn.Module:
        """Trả về LSTM model."""
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
                    nn.Linear(64, 1)
                )

            def forward(self, x):
                # x: (batch, seq_len, input_size)
                lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
                last_hidden = lstm_out[:, -1, :]  # take last timestep
                out = self.fc(last_hidden)  # (batch, 1)
                return out.squeeze(-1)

        return LSTMNet(input_size)
