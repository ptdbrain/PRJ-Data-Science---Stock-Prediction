"""
Predict giá TCB dùng best model.
═════════════════════════════════
Chạy: python -m models.predict

Load best model → predict giá cho mỗi ngày trong test set → lưu predictions table.
(Phase 1: predict trên data tĩnh, Phase 2 sẽ predict ngày tiếp theo)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from database.connection import read_table, write_table, get_connection
from config.settings import LOOKBACK_DAYS, ALL_FEATURES

from models.lstm_model import LSTMPredictor
from models.gru_model import GRUPredictor
from models.transformer_model import TransformerPredictor

logger.add("logs/predict.log", rotation="1 week")

MODEL_MAP = {
    'lstm': LSTMPredictor,
    'gru': GRUPredictor,
    'transformer': TransformerPredictor,
}


def get_best_model_name() -> str:
    """Tìm best model từ model_metrics table."""
    metrics = read_table("model_metrics")
    best = metrics[metrics['is_best'] == 1]
    if best.empty:
        best = metrics.sort_values('mape').head(1)
    return best.iloc[0]['model_name']


def predict_all():
    """
    Load best model, predict trên toàn bộ data, lưu predictions.
    Phase 1: dùng data tĩnh, predict trên test period.
    """
    best_name = get_best_model_name()
    logger.info(f"Sử dụng best model: {best_name}")

    # Load model
    model_class = MODEL_MAP[best_name]
    model = model_class()
    model.load()

    # Load data
    df = read_table("merged_features")
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    # Predict cho từng ngày (sliding window)
    predictions = []
    for i in range(LOOKBACK_DAYS, len(df) - 1):
        window = df.iloc[i - LOOKBACK_DAYS:i]
        pred_price = model.predict_next(window)
        actual_price = df.iloc[i]['target']  # Giá ngày hôm sau

        predictions.append({
            'date': df.iloc[i + 1]['date'],  # Ngày được predict
            'model_name': best_name,
            'predicted_price': round(pred_price, 0),
            'actual_price': round(actual_price, 0),
            'predicted_at': datetime.now().isoformat()
        })

    pred_df = pd.DataFrame(predictions)
    write_table(pred_df, "predictions")

    logger.info(f"✅ Đã lưu {len(pred_df)} predictions vào database")

    # Thống kê nhanh
    errors = np.abs(pred_df['predicted_price'] - pred_df['actual_price'])
    logger.info(f"  MAE: {errors.mean():,.0f} VND")
    logger.info(f"  Max error: {errors.max():,.0f} VND")


if __name__ == "__main__":
    predict_all()
