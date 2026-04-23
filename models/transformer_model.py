"""
Transformer Model cho dự đoán giá TCB.
══════════════════════════════════════
Phụ trách: Thành viên E
Branch: feature/model-transformer
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
        """Trả về Transformer Encoder model.

        Kiến trúc:
            Input → Linear Projection (input_size → d_model) →
            Positional Encoding →
            TransformerEncoder (NUM_LAYERS lớp, nhead=8) →
            Output của timestep cuối →
            FC(d_model → 64) → ReLU → FC(64 → 1)

        Lưu ý:
        - d_model phải chia hết cho nhead. HIDDEN_SIZE=128, nhead=8 → 128/8=16 ✅
        - Dùng causal mask để tránh model nhìn tương lai (tùy chọn)
        """
        class PositionalEncoding(nn.Module):
            """Positional Encoding chuẩn từ paper 'Attention is All You Need'."""
            def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                # Tạo ma trận PE: (max_len, d_model)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)  # (1, max_len, d_model)
                self.register_buffer('pe', pe)

            def forward(self, x):
                # x: (batch, seq_len, d_model)
                x = x + self.pe[:, :x.size(1), :]
                return self.dropout(x)

        class TransformerNet(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                # Đảm bảo d_model chia hết cho nhead=8
                self.d_model = HIDDEN_SIZE  # 128
                self.nhead = 8

                # Projection: input_size → d_model
                self.input_proj = nn.Linear(input_size, self.d_model)

                # Positional Encoding
                self.pos_encoding = PositionalEncoding(self.d_model, dropout=DROPOUT)

                # Transformer Encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.d_model * 4,  # 512
                    dropout=DROPOUT,
                    batch_first=True,
                    norm_first=True,    # Pre-LayerNorm → ổn định hơn khi train
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=NUM_LAYERS
                )

                # Output head
                self.fc = nn.Sequential(
                    nn.Linear(self.d_model, 64),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(64, 1)
                )

            def forward(self, x):
                # x: (batch, seq_len, input_size)
                x = self.input_proj(x)      # → (batch, seq_len, d_model)
                x = self.pos_encoding(x)    # thêm positional info
                x = self.transformer(x)     # → (batch, seq_len, d_model)
                x = x[:, -1, :]            # lấy output timestep cuối
                return self.fc(x).squeeze(-1)  # → (batch,)

        return TransformerNet(input_size)
