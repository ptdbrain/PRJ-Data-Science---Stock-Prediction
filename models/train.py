"""
Train + so sánh tất cả models.
══════════════════════════════
Chạy: python -m models.train

Train 3 models (LSTM, GRU, Transformer), so sánh kết quả,
lưu metrics vào model_metrics table, đánh dấu best model.
"""
import pandas as pd
from utils.logger import logger
from database.connection import read_table, write_table, get_connection
from config.settings import ALL_FEATURES

from models.lstm_model import LSTMPredictor
from models.gru_model import GRUPredictor
from models.transformer_model import TransformerPredictor

logger.add("logs/train.log", rotation="1 week")


def get_feature_cols(df: pd.DataFrame) -> list:
    """Lấy danh sách feature columns có trong data."""
    available = [c for c in ALL_FEATURES if c in df.columns]
    logger.info(f"Features available: {len(available)}/{len(ALL_FEATURES)}")
    return available


def train_all_models():
    """Train tất cả models và so sánh."""
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TRAINING TẤT CẢ MODELS")
    logger.info("=" * 60)

    # Load data
    df = read_table("merged_features")
    feature_cols = get_feature_cols(df)

    if len(df) < 100:
        logger.error(f"Chỉ có {len(df)} rows — cần ít nhất 100. "
                      f"Chạy preprocessing trước.")
        return

    # Train từng model
    models = [LSTMPredictor(), GRUPredictor(), TransformerPredictor()]
    all_metrics = []

    for model in models:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training: {model.model_name}")
        logger.info(f"{'='*40}")
        try:
            metrics = model.fit(df, feature_cols)
            model.save()
            all_metrics.append(metrics)
        except NotImplementedError:
            logger.warning(f"⏭️  {model.model_name} chưa implement — bỏ qua")
        except Exception as e:
            logger.error(f"❌ {model.model_name} lỗi: {e}")

    if not all_metrics:
        logger.error("Không có model nào train thành công!")
        return

    # So sánh và chọn best model
    metrics_df = pd.DataFrame(all_metrics)

    # Best = MAPE thấp nhất
    best_idx = metrics_df['mape'].idxmin()
    metrics_df['is_best'] = 0
    metrics_df.loc[best_idx, 'is_best'] = 1

    # Lưu metrics vào DB
    write_table(metrics_df, "model_metrics")

    # In kết quả so sánh
    logger.info(f"\n{'='*60}")
    logger.info("KẾT QUẢ SO SÁNH")
    logger.info(f"{'='*60}")
    for _, row in metrics_df.iterrows():
        best_tag = " ⭐ BEST" if row['is_best'] else ""
        logger.info(
            f"  {row['model_name']:<15} | "
            f"RMSE: {row['rmse']:>10,.0f} | "
            f"MAPE: {row['mape']:>6.2f}% | "
            f"Direction: {row['directional_accuracy']:>5.1f}%"
            f"{best_tag}"
        )

    best_name = metrics_df.loc[best_idx, 'model_name']
    logger.info(f"\n🏆 Best model: {best_name}")


if __name__ == "__main__":
    train_all_models()
