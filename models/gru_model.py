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
        """Trả về GRU model."""

        # TODO: Thành viên E implement
        # Tương tự LSTM nhưng dùng nn.GRU thay vì nn.LSTM
        # GRU thường train nhanh hơn LSTM, ít parameters hơn

        raise NotImplementedError("Thành viên E cần implement")
