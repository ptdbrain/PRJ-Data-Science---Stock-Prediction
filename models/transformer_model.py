"""
Transformer Model cho dự đoán xu hướng giá TCB (Classification).
══════════════════════════════════════════════════════════════════
Kiến trúc:
    Input → Linear Projection (input_size → d_model) →
    Positional Encoding →
    TransformerEncoder (NUM_LAYERS lớp, nhead=8) →
    Output của timestep cuối →
    FC(d_model → 64) → ReLU → FC(64 → 1)  [raw logit]

Lưu ý:
- d_model phải chia hết cho nhead. HIDDEN_SIZE=128, nhead=8 → 128/8=16 ✅
- BCEWithLogitsLoss được áp dụng trong BasePredictor.fit (không dùng Sigmoid ở đây)
"""
import math
import torch
import torch.nn as nn
from models.base_model import BasePredictor
from config.settings import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class TransformerPredictor(BasePredictor):

    def __init__(self, **kwargs):
        super().__init__(model_name="transformer", **kwargs)

    def build_model(self, input_size: int) -> nn.Module:
        """Trả về Transformer Encoder model với output là raw logit."""

        class PositionalEncoding(nn.Module):
            """Positional Encoding chuẩn từ paper 'Attention is All You Need'."""
            def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:, :x.size(1), :]
                return self.dropout(x)

        class TransformerNet(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.d_model = HIDDEN_SIZE  # 128
                self.nhead = 8

                self.input_proj = nn.Linear(input_size, self.d_model)
                self.pos_encoding = PositionalEncoding(self.d_model, dropout=DROPOUT)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.d_model * 4,
                    dropout=DROPOUT,
                    batch_first=True,
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

                self.fc = nn.Sequential(
                    nn.Linear(self.d_model, 64),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                x = self.input_proj(x)
                x = self.pos_encoding(x)
                x = self.transformer(x)
                x = x[:, -1, :]  # lấy output timestep cuối
                return self.fc(x).squeeze(-1)  # raw logit

        return TransformerNet(input_size)
