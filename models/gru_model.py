"""
GRU Model cho dự đoán giá TCB.
══════════════════════════════
Phụ trách: Thành viên E
Branch: feature/model-gru
"""
import torch
import torch.nn as nn
from models.base_model import BasePredictor
from config.settings import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class GRUPredictor(BasePredictor):

    def __init__(self):
        super().__init__(model_name="gru")

    def build_model(self, input_size: int) -> nn.Module:
        """Trả về GRU model.

        GRU (Gated Recurrent Unit) — biến thể nhẹ hơn LSTM:
        - Ít parameters hơn (~75% so với LSTM cùng kích thước)
        - Train nhanh hơn (~20-30%)
        - Thường hiệu quả tương đương hoặc tốt hơn trên chuỗi ngắn

        Kiến trúc:
            Input → GRU(2 layers, hidden=128) → FC(128→64) → ReLU → Dropout → FC(64→1)
        """
        class GRUNet(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.gru = nn.GRU(
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
                gru_out, _ = self.gru(x)         # (batch, seq_len, hidden_size)
                last_hidden = gru_out[:, -1, :]  # lấy timestep cuối
                out = self.fc(last_hidden)        # (batch, 1)
                return out.squeeze(-1)

        return GRUNet(input_size)
