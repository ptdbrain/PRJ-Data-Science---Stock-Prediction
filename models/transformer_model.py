"""
Transformer Model cho dự đoán giá TCB.
══════════════════════════════════════
Phụ trách: Thành viên E
Branch: feature/model-transformer
"""
import torch
import torch.nn as nn
import math
from models.base_model import BasePredictor
from config.settings import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class TransformerPredictor(BasePredictor):

    def __init__(self):
        super().__init__(model_name="transformer")

    def build_model(self, input_size: int) -> nn.Module:
        """Trả về Transformer model."""

        # TODO: Thành viên E implement
        # Gợi ý kiến trúc:
        #
        # class TransformerNet(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.input_proj = nn.Linear(input_size, HIDDEN_SIZE)
        #         self.pos_encoding = PositionalEncoding(HIDDEN_SIZE)
        #         encoder_layer = nn.TransformerEncoderLayer(
        #             d_model=HIDDEN_SIZE, nhead=8,
        #             dim_feedforward=256, dropout=DROPOUT, batch_first=True
        #         )
        #         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        #         self.fc = nn.Sequential(
        #             nn.Linear(HIDDEN_SIZE, 64),
        #             nn.ReLU(),
        #             nn.Linear(64, 1)
        #         )
        #
        #     def forward(self, x):
        #         x = self.input_proj(x)
        #         x = self.pos_encoding(x)
        #         x = self.transformer(x)
        #         x = x[:, -1, :]     # Lấy output timestep cuối
        #         return self.fc(x).squeeze(-1)
        #
        # Cần thêm class PositionalEncoding (google "pytorch positional encoding")

        raise NotImplementedError("Thành viên E cần implement")
