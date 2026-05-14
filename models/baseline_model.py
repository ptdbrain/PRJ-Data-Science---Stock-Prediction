"""
Baseline Models cho dự đoán xu hướng giá TCB.
══════════════════════════════════════════════
Gồm 2 baseline cổ điển để so sánh với LSTM/GRU/Transformer:
  1. Logistic Regression  — tuyến tính, dễ giải thích
  2. Random Forest        — phi tuyến, robust với feature noise

Không cần GPU. Train/predict rất nhanh.
Dùng TimeSeriesSplit (không shuffle) để tránh look-ahead bias.

Chạy:
    python -m models.baseline_model
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from utils.logger import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

from config.settings import (
    ALL_FEATURES,
    LOOKBACK_DAYS,
    MODEL_DIR,
    TRAIN_RATIO,
    TREND_THRESHOLD,
    VAL_RATIO,
)
from database.connection import read_table, write_table

logger.add("logs/baseline.log", rotation="1 week")


# ================================================================
# Feature Engineering for Flat Models
# ================================================================
def _flatten_sequences(features_scaled: np.ndarray, target: np.ndarray, lookback: int):
    """
    Chuyển chuỗi thời gian thành ma trận flat cho sklearn.

    Thay vì dùng 3D tensor (samples, timesteps, features),
    chúng ta flatten thành (samples, timesteps * features).

    Đây là cách tiếp cận phổ biến cho Logistic Regression / Random Forest
    với chuỗi thời gian ngắn.
    """
    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        window = features_scaled[i - lookback:i].flatten()  # flatten (T, F) → (T*F,)
        X.append(window)
        y.append(target[i])
    return np.array(X), np.array(y)


def _evaluate_baseline(model_name: str, y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Tính classification metrics cho baseline model."""
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    logger.info(
        f"[{model_name}] Acc: {acc:.1f}% | Precision: {prec:.4f} | "
        f"Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
    )
    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "test_loss": float(np.mean((y_prob - y_test.astype(float)) ** 2)),
    }


def _find_optimal_threshold_baseline(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find validation threshold while avoiding one-class predictions."""
    best_score, best_f1, best_t = -1.0, -1.0, 0.5
    for t in np.arange(0.30, 0.71, 0.01):
        preds = (y_prob >= t).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        if (score, f1) > (best_score, best_f1):
            best_score = float(score)
            best_f1 = f1
            best_t = float(t)
    return best_t


# ================================================================
# Main Baseline Training
# ================================================================
def train_baselines(df: pd.DataFrame, feature_cols: list) -> list:
    """
    Train Logistic Regression + Random Forest trên merged_features.

    Sử dụng:
      - Chronological train/test split (không shuffle)
      - MinMaxScaler fit chỉ trên train set
      - Lookback window giống LSTM (flattened)

    Trả về list[dict] metrics cho từng baseline model.
    """
    logger.info("=" * 60)
    logger.info("TRAINING BASELINE MODELS")
    logger.info("=" * 60)

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    features = df[feature_cols].values
    target = df["target"].values.astype(int)

    raw_count = len(df)
    train_end = int(raw_count * TRAIN_RATIO)
    val_end = int(raw_count * (TRAIN_RATIO + VAL_RATIO))

    logger.info(
        f"  Dataset: {raw_count} rows | "
        f"Train: {train_end} | Val: {val_end - train_end} | Test: {raw_count - val_end}"
    )
    logger.info(f"  Lookback: {LOOKBACK_DAYS} ngày | Features/sample (flattened): {LOOKBACK_DAYS * len(feature_cols)}")

    # Scale features (fit on train only)
    scaler = RobustScaler()
    scaler.fit(features[:train_end])
    features_scaled = scaler.transform(features)

    # Tạo sequences
    X_all, y_all = _flatten_sequences(features_scaled, target, LOOKBACK_DAYS)

    # Tính lại split index sau khi trừ lookback
    # Sequence thứ i tương ứng với raw_index = i + LOOKBACK_DAYS
    train_end_seq = train_end - LOOKBACK_DAYS
    val_end_seq = val_end - LOOKBACK_DAYS

    X_train, y_train = X_all[:train_end_seq], y_all[:train_end_seq]
    X_val, y_val = X_all[train_end_seq:val_end_seq], y_all[train_end_seq:val_end_seq]
    X_test, y_test = X_all[val_end_seq:], y_all[val_end_seq:]

    logger.info(f"  Sequences -- Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    pos_ratio = y_train.mean()
    logger.info(f"  Class balance (train): {pos_ratio:.1%} UP / {1-pos_ratio:.1%} DOWN")

    all_metrics = []

    # ── Logistic Regression ──────────────────────────────────────
    logger.info("\n[Logistic Regression]")
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",  # xử lý class imbalance
    )
    lr.fit(X_train, y_train)
    lr_prob_val = lr.predict_proba(X_val)[:, 1]
    lr_threshold = _find_optimal_threshold_baseline(y_val, lr_prob_val)
    logger.info(f"  Optimal threshold (val balanced accuracy): {lr_threshold:.3f}")
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= lr_threshold).astype(int)
    metrics_lr = _evaluate_baseline("logistic_regression", y_test, lr_pred, lr_prob)
    metrics_lr["trained_at"] = datetime.now().isoformat()
    metrics_lr["threshold"] = lr_threshold
    all_metrics.append(metrics_lr)

    # Lưu model
    _save_sklearn_model(lr, scaler, feature_cols, "logistic_regression", metrics_lr)

    # ── Random Forest ─────────────────────────────────────────────
    logger.info("\n[Random Forest]")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_prob_val = rf.predict_proba(X_val)[:, 1]
    rf_threshold = _find_optimal_threshold_baseline(y_val, rf_prob_val)
    logger.info(f"  Optimal threshold (val balanced accuracy): {rf_threshold:.3f}")
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= rf_threshold).astype(int)
    metrics_rf = _evaluate_baseline("random_forest", y_test, rf_pred, rf_prob)
    metrics_rf["trained_at"] = datetime.now().isoformat()
    metrics_rf["threshold"] = rf_threshold
    all_metrics.append(metrics_rf)

    _save_sklearn_model(rf, scaler, feature_cols, "random_forest", metrics_rf)

    logger.info("=" * 60)
    return all_metrics


def _save_sklearn_model(clf, scaler, feature_cols: list, name: str, metrics: dict):
    """Lưu sklearn model + metadata ra file JSON (không cần torch)."""
    import pickle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"tcb_{name}.pkl"
    meta_path = MODEL_DIR / f"tcb_{name}_meta.json"

    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler}, f)

    meta = {
        "model_name": name,
        "feature_cols": feature_cols,
        "lookback_days": LOOKBACK_DAYS,
        "threshold": metrics.get("threshold", TREND_THRESHOLD),
        "saved_at": datetime.now().isoformat(),
        **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"  Saved: {model_path.name}")


def load_sklearn_model(name: str):
    """Load sklearn model và scaler từ file pickle."""
    import pickle
    path = MODEL_DIR / f"tcb_{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model '{name}' không tìm thấy: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]


def predict_baseline(name: str, df: pd.DataFrame, feature_cols: list) -> float:
    """Trả về P(tăng) dùng baseline model."""
    clf, scaler = load_sklearn_model(name)
    features = df[feature_cols].values[-LOOKBACK_DAYS:]
    features_scaled = scaler.transform(features)
    X = features_scaled.flatten().reshape(1, -1)
    prob = clf.predict_proba(X)[0][1]
    return float(prob)


if __name__ == "__main__":
    df = read_table("merged_features")
    if df.empty:
        logger.error("merged_features trống — chạy preprocessing trước.")
    else:
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]
        metrics_list = train_baselines(df, feature_cols)

        metrics_df = pd.DataFrame(metrics_list)
        # Đánh dấu best
        best_idx = metrics_df["f1"].idxmax()
        metrics_df["is_best"] = 0
        metrics_df.loc[best_idx, "is_best"] = 1

        # Lưu vào DB (append hoặc replace)
        from database.connection import get_connection
        conn = get_connection()
        for _, row in metrics_df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO model_metrics
                (model_name, accuracy, precision, recall, f1, roc_auc, test_loss, trained_at, is_best)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["model_name"], row["accuracy"], row.get("precision"),
                    row.get("recall"), row["f1"], row.get("roc_auc"),
                    row.get("test_loss"), row.get("trained_at"), int(row.get("is_best", 0)),
                ),
            )
        conn.commit()
        conn.close()
        logger.info(f"Đã lưu {len(metrics_df)} baseline metrics vào model_metrics.")
