"""
Predict xu hướng TCB dùng best model (Classification).
═══════════════════════════════════════════════════════
Chạy: python -m models.predict

Thay vì dự đoán giá cụ thể, bây giờ dự đoán:
  - predicted_proba : P(giá ngày mai TĂNG) ∈ [0.0, 1.0]
  - predicted_trend : 1 (Tăng) hoặc 0 (Giảm/Đi ngang)

Các hàm:
  - predict_all()  : predict trên toàn bộ merged_features, lưu DB
  - get_latest_prediction() : lấy dự đoán mới nhất cho ngày hôm sau
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from utils.logger import logger
from database.connection import read_table, write_table, get_connection, table_exists
from config.settings import ALL_FEATURES, DEFAULT_MODEL_NAME, TREND_THRESHOLD

from models.lstm_model import LSTMPredictor
from models.gru_model import GRUPredictor
from models.transformer_model import TransformerPredictor

logger.add("logs/predict.log", rotation="1 week")

DEEP_MODEL_MAP = {
    "lstm": LSTMPredictor,
    "gru": GRUPredictor,
    "transformer": TransformerPredictor,
}

SKLEARN_MODEL_NAMES = {"logistic_regression", "random_forest"}


def get_best_model_name() -> str:
    """Tìm best model từ model_metrics table (dựa theo Accuracy)."""
    metrics = read_table("model_metrics")
    if not metrics.empty:
        # Ưu tiên Accuracy (F1 misleading khi model predict tất cả UP)
        sort_col = "accuracy" if "accuracy" in metrics.columns else "f1"
        best_rows = metrics[metrics.get("is_best", pd.Series(0)) == 1]
        if not best_rows.empty:
            return best_rows.iloc[0]["model_name"]
        return metrics.sort_values(sort_col, ascending=False).iloc[0]["model_name"]

    # Fallback: saved metadata files
    logger.warning("model_metrics table empty — attempting fallback to saved metadata files")
    saved_dir = Path(__file__).resolve().parents[0] / "saved"
    if saved_dir.exists():
        candidates = []
        for path in list(saved_dir.glob("tcb_*_meta.json")):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                acc = data.get("accuracy", 0)
                name = data.get("model_name") or path.stem.split("_", 1)[-1]
                candidates.append((float(acc), name))
            except Exception:
                continue
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            chosen = candidates[0][1]
            logger.warning(f"Fallback to saved metadata: {chosen}")
            return chosen

    logger.warning(f"No saved metrics — falling back to DEFAULT_MODEL_NAME={DEFAULT_MODEL_NAME}")
    return DEFAULT_MODEL_NAME


def _predict_with_deep_model(model_name: str, df: pd.DataFrame, feature_cols: list) -> list:
    """Predict dùng LSTM/GRU/Transformer."""
    model_class = DEEP_MODEL_MAP.get(model_name)
    if model_class is None:
        raise ValueError(f"No deep model implementation for '{model_name}'")

    model = model_class()
    model.load(name=model_name)
    lookback = model.lookback_days
    threshold = getattr(model, "threshold", TREND_THRESHOLD)  # Optimal threshold from training
    logger.info(f"  Using threshold: {threshold:.3f}")

    predictions = []
    for i in range(lookback, len(df) - 1):
        window = df.iloc[i - lookback:i]
        proba = model.predict_proba(window)
        trend = int(proba >= threshold)
        actual_trend = int(df.iloc[i]["target"]) if "target" in df.columns else None

        predictions.append({
            "date": df.iloc[i + 1]["date"],
            "model_name": model_name,
            "predicted_proba": round(proba, 6),
            "predicted_trend": trend,
            "actual_trend": actual_trend,
            "predicted_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        })

    return predictions


def _predict_with_sklearn_model(model_name: str, df: pd.DataFrame, feature_cols: list) -> list:
    """Predict dùng Logistic Regression hoac Random Forest."""
    from models.baseline_model import load_sklearn_model, LOOKBACK_DAYS
    
    clf, scaler = load_sklearn_model(model_name)

    # Load optimal threshold from metadata
    from config.settings import MODEL_DIR
    meta_path = MODEL_DIR / f"tcb_{model_name}_meta.json"
    threshold = TREND_THRESHOLD  # fallback
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
            threshold = meta.get("threshold", TREND_THRESHOLD)
    logger.info(f"  Using threshold: {threshold:.3f}")

    features = df[feature_cols].values
    features_scaled = scaler.transform(features)

    predictions = []
    for i in range(LOOKBACK_DAYS, len(df) - 1):
        window = features_scaled[i - LOOKBACK_DAYS:i].flatten().reshape(1, -1)
        proba = float(clf.predict_proba(window)[0][1])
        trend = int(proba >= threshold)
        actual_trend = int(df.iloc[i]["target"]) if "target" in df.columns else None

        predictions.append({
            "date": df.iloc[i + 1]["date"],
            "model_name": model_name,
            "predicted_proba": round(proba, 6),
            "predicted_trend": trend,
            "actual_trend": actual_trend,
            "predicted_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        })

    return predictions


def predict_all():
    """
    Load best model, predict xu hướng trên toàn bộ data, lưu predictions.
    """
    best_name = get_best_model_name()
    logger.info(f"Sử dụng best model: {best_name}")

    df = read_table("merged_features")
    if df.empty:
        logger.error("❌ merged_features trống! Chạy preprocessing trước.")
        return

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    try:
        if best_name in SKLEARN_MODEL_NAMES:
            predictions = _predict_with_sklearn_model(best_name, df, feature_cols)
        else:
            predictions = _predict_with_deep_model(best_name, df, feature_cols)
    except Exception as e:
        logger.error(f"❌ Prediction thất bại với {best_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    if not predictions:
        logger.warning("Không tạo được prediction nào.")
        return

    pred_df = pd.DataFrame(predictions)
    write_table(pred_df, "predictions", if_exists="replace")
    logger.info(f"✅ Đã lưu {len(pred_df)} predictions vào database")

    # Thống kê nhanh
    known = pred_df.dropna(subset=["actual_trend"])
    if not known.empty:
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(known["actual_trend"], known["predicted_trend"]) * 100
        f1 = f1_score(known["actual_trend"], known["predicted_trend"], zero_division=0)
        up_pred = known["predicted_trend"].mean() * 100
        logger.info(f"  Accuracy (known): {acc:.1f}%")
        logger.info(f"  F1 (known):       {f1:.4f}")
        logger.info(f"  % Dự đoán Tăng:  {up_pred:.1f}%")


def get_latest_prediction() -> dict:
    """
    Trả về dự đoán xu hướng cho ngày tiếp theo.
    Dùng trong Streamlit để hiển thị dự đoán mới nhất.
    """
    if not table_exists("predictions"):
        return {}

    preds = read_table("predictions")
    if preds.empty:
        return {}

    preds = preds.sort_values("date")
    latest = preds.iloc[-1]

    return {
        "date": latest.get("date"),
        "model_name": latest.get("model_name"),
        "predicted_proba": float(latest.get("predicted_proba", 0.5)),
        "predicted_trend": int(latest.get("predicted_trend", 0)),
        "trend_label": "📈 TĂNG" if int(latest.get("predicted_trend", 0)) == 1 else "📉 GIẢM",
    }


if __name__ == "__main__":
    predict_all()
