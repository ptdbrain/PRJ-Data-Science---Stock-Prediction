"""
Train + so sánh tất cả models (Classification).
═════════════════════════════════════════════════
Chạy: python -m models.train

Train 5 models:
  Baseline: Logistic Regression, Random Forest
  Deep:     LSTM, GRU, Transformer
So sánh theo F1-score (phù hợp với classification + imbalanced data).
"""
import pandas as pd
from utils.logger import logger
from database.connection import read_table, write_table
from config.settings import ALL_FEATURES

from models.lstm_model import LSTMPredictor
from models.gru_model import GRUPredictor
from models.transformer_model import TransformerPredictor
from models.baseline_model import train_baselines

logger.add("logs/train.log", rotation="1 week")


def get_feature_cols(df: pd.DataFrame) -> list:
    """Lấy danh sách feature columns có trong data."""
    available = [c for c in ALL_FEATURES if c in df.columns]
    logger.info(f"Features available: {len(available)}/{len(ALL_FEATURES)}")
    if len(available) < len(ALL_FEATURES):
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        logger.warning(f"Thiếu features: {missing}")
    return available


def train_all_models():
    """Train tất cả models và chọn best theo ROC-AUC."""
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TRAINING TẤT CẢ MODELS (Classification)")
    logger.info("=" * 60)

    # Load data
    df = read_table("merged_features")

    if df.empty or len(df) < 100:
        logger.error(
            f"Chỉ có {len(df)} rows — cần ít nhất 100. "
            "Chạy preprocessing trước."
        )
        return

    feature_cols = get_feature_cols(df)

    # Kiểm tra target là binary
    target_vals = df["target"].unique()
    if not set(target_vals).issubset({0, 1}):
        logger.error(
            f"Target không phải binary (0/1): {target_vals}. "
            "Chạy lại merge_features để tạo target mới."
        )
        return

    up_ratio = df["target"].mean()
    logger.info(f"Class balance: {up_ratio:.1%} UP / {1-up_ratio:.1%} DOWN")

    all_metrics = []

    # ── Phase 1: Baseline Models ─────────────────────────────────
    logger.info("\n📊 PHASE 1: BASELINE MODELS")
    try:
        baseline_metrics = train_baselines(df, feature_cols)
        all_metrics.extend(baseline_metrics)
    except Exception as e:
        logger.error(f"❌ Baseline models lỗi: {e}")

    # ── Phase 2: Deep Learning Models ───────────────────────────
    logger.info("\n🧠 PHASE 2: DEEP LEARNING MODELS")
    deep_models = [LSTMPredictor(), GRUPredictor(), TransformerPredictor()]

    for model in deep_models:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Training: {model.model_name}")
        logger.info(f"{'=' * 40}")
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

    # ── So sánh và chọn best model ───────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)

    # Chuẩn hoá: thêm cột thiếu nếu có
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "is_best"]:
        if col not in metrics_df.columns:
            metrics_df[col] = None

    # Best = ROC-AUC cao nhất. Accuracy dễ bị đánh lừa khi test set lệch class
    # hoặc model đoán gần như toàn DOWN/UP.
    valid_auc = metrics_df["roc_auc"].dropna()
    if not valid_auc.empty:
        best_idx = valid_auc.idxmax()
        metrics_df["is_best"] = 0
        metrics_df.loc[best_idx, "is_best"] = 1
    else:
        metrics_df["is_best"] = 0

    # Lưu metrics vào DB
    write_table(metrics_df, "model_metrics", if_exists="replace")

    # In kết quả
    logger.info(f"\n{'=' * 75}")
    logger.info("KET QUA SO SANH -- CLASSIFICATION")
    logger.info(f"{'=' * 75}")
    logger.info(f"{'Model':<25} {'Acc (%)':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Best':>6}")
    logger.info("-" * 75)

    for _, row in metrics_df.sort_values("roc_auc", ascending=False).iterrows():
        best_tag = "***" if row.get("is_best") else ""
        logger.info(
            f"  {row['model_name']:<23} "
            f"{row.get('accuracy', 0):>9.1f} "
            f"{row.get('precision', 0):>10.4f} "
            f"{row.get('recall', 0):>8.4f} "
            f"{row.get('f1', 0):>8.4f} "
            f"{row.get('roc_auc', 0):>8.4f} "
            f"{best_tag:>6}"
        )

    best_name = metrics_df.loc[metrics_df["is_best"] == 1, "model_name"].values
    if len(best_name) > 0:
        logger.info(f"\nBest model (by ROC-AUC): {best_name[0]}")


if __name__ == "__main__":
    train_all_models()
