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

    def __init__(self):
        super().__init__(model_name="lstm")

    def build_model(self, input_size: int) -> nn.Module:
        """Trả về LSTM model."""

        # TODO: Thành viên D implement
        # Gợi ý kiến trúc:
        #
        # class LSTMNet(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.lstm = nn.LSTM(
        #             input_size=input_size,
        #             hidden_size=HIDDEN_SIZE,
        #             num_layers=NUM_LAYERS,
        #             batch_first=True,
        #             dropout=DROPOUT
        #         )
        #         self.fc = nn.Sequential(
        #             nn.Linear(HIDDEN_SIZE, 64),
        #             nn.ReLU(),
        #             nn.Dropout(DROPOUT),
        #             nn.Linear(64, 1)
        #         )
        #
        #     def forward(self, x):
        #         lstm_out, _ = self.lstm(x)
        #         last_hidden = lstm_out[:, -1, :]  # Lấy output timestep cuối
        #         return self.fc(last_hidden).squeeze(-1)
        #
        # Nâng cao: thêm Attention mechanism
        # (xem reference: models/base_model.py phần comment)

        raise NotImplementedError("Thành viên D cần implement")
