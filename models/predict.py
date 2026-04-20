"""
Predict giá TCB dùng best model.
═════════════════════════════════
Chạy: python -m models.predict

Load best model → predict giá cho mỗi ngày trong test set → lưu predictions table.

Các hàm:
  - predict_all(): predict trên toàn bộ merged_features, lưu DB
  - update_actual_prices(): cập nhật actual_price sau khi biết giá thực tế
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from database.connection import read_table, write_table, get_connection, table_exists
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
        # target = close ngày hôm sau (do merge_features đã shift -1)
        actual_price = df.iloc[i]['target']

        error_pct = None
        if actual_price and actual_price != 0:
            error_pct = round(abs(pred_price - actual_price) / actual_price * 100, 4)

        predictions.append({
            'date': df.iloc[i + 1]['date'],  # Ngày được predict
            'model_name': best_name,
            'predicted_price': round(pred_price, 0),
            'actual_price': round(actual_price, 0) if actual_price else None,
            'error_pct': error_pct,
            'predicted_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        })

    pred_df = pd.DataFrame(predictions)
    write_table(pred_df, "predictions")

    logger.info(f"✅ Đã lưu {len(pred_df)} predictions vào database")

    # Thống kê nhanh
    known = pred_df.dropna(subset=['actual_price', 'predicted_price'])
    if not known.empty:
        errors = np.abs(known['predicted_price'] - known['actual_price'])
        logger.info(f"  MAE: {errors.mean():,.0f} VND")
        logger.info(f"  Max error: {errors.max():,.0f} VND")
        if 'error_pct' in known.columns:
            logger.info(f"  MAPE: {known['error_pct'].mean():.2f}%")


def update_actual_prices():
    """
    Cập nhật actual_price và error_pct cho các predictions đã có giá thực tế.

    Chạy hàm này sau khi collect_prices đã fetch giá mới,
    để cập nhật các dự đoán trước đó khả chưa có actual.

    Ví dụ:
        # Sau khi chạy collect_prices vào buổi tối:
        from models.predict import update_actual_prices
        update_actual_prices()
    """
    if not table_exists('predictions') or not table_exists('raw_prices'):
        logger.warning("predictions hoặc raw_prices chưa tồn tại — bỏ qua.")
        return

    preds = read_table('predictions')
    prices = read_table('raw_prices')[['date', 'close']].rename(columns={'close': 'actual_close'})

    # Join predictions với giá thực tế
    merged = preds.merge(prices, on='date', how='left')

    updated_count = 0
    conn = get_connection()
    for _, row in merged.iterrows():
        if pd.isna(row['actual_price']) and not pd.isna(row.get('actual_close')):
            actual = float(row['actual_close'])
            pred = float(row['predicted_price'])
            error_pct = round(abs(pred - actual) / actual * 100, 4) if actual != 0 else None

            conn.execute(
                """
                UPDATE predictions
                SET actual_price = ?, error_pct = ?, updated_at = ?
                WHERE date = ? AND model_name = ?
                """,
                (round(actual, 0), error_pct, datetime.now().isoformat(),
                 row['date'], row['model_name'])
            )
            updated_count += 1

    conn.commit()
    conn.close()

    if updated_count:
        logger.info(f"✅ Cập nhật actual_price cho {updated_count} predictions")
    else:
        logger.info("Không có prediction nào cần cập nhật actual_price")


if __name__ == "__main__":
    predict_all()
