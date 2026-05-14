"""
GRU Model cho dự đoán xu hướng giá TCB (Classification).
══════════════════════════════════════════════════════════
GRU (Gated Recurrent Unit) — biến thể nhẹ hơn LSTM:
  - Ít parameters hơn (~75% so với LSTM cùng kích thước)
  - Train nhanh hơn (~20-30%)
  - Thường hiệu quả tương đương hoặc tốt hơn trên chuỗi ngắn

Output: raw logit (BCEWithLogitsLoss được áp dụng trong BasePredictor.fit)
"""
import torch
import torch.nn as nn
from models.base_model import BasePredictor
from config.settings import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class GRUPredictor(BasePredictor):

    def __init__(self, **kwargs):
        super().__init__(model_name="gru", **kwargs)

    def build_model(self, input_size: int) -> nn.Module:
        """Trả về GRU model với output là raw logit (1 neuron).

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
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                gru_out, _ = self.gru(x)
                last_hidden = gru_out[:, -1, :]
                return self.fc(last_hidden).squeeze(-1)  # raw logit

        return GRUNet(input_size)
