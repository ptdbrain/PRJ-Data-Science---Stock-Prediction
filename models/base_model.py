"""
Base class cho tất cả prediction models (Classification version).
════════════════════════════════════════════════════════════════
Chuyển đổi từ Regression (dự đoán giá) sang Classification (dự đoán xu hướng).

Thay đổi chính:
  - Loss function: MSELoss → BCELoss
  - Output: giá trị liên tục → xác suất P(tăng) ∈ [0, 1]
  - Metrics: RMSE/MAPE → Accuracy, Precision, Recall, F1, ROC-AUC
"""
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.logger import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from config.settings import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    LOOKBACK_DAYS,
    MODEL_DIR,
    TRAIN_RATIO,
    TREND_THRESHOLD,
    VAL_RATIO,
)


class BasePredictor(ABC):
    """Base predictor shared by LSTM/GRU/Transformer models (Classification)."""

    def __init__(
        self,
        model_name: str,
        *,
        lookback_days: int = LOOKBACK_DAYS,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        device=DEVICE,
        threshold: float = TREND_THRESHOLD,
    ):
        self.model_name = model_name
        self.lookback_days = lookback_days
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.device = device
        self.threshold = threshold

        self.model: nn.Module | None = None
        self.feature_cols: list = []
        self.history = {"train_loss": [], "val_loss": []}
        self.split_metadata: dict = {}

        # Feature scaler (RobustScaler) — resistant to outliers in financial data
        from sklearn.preprocessing import RobustScaler
        self.feature_scaler = RobustScaler()

    @abstractmethod
    def build_model(self, input_size: int) -> nn.Module:
        """Subclass must return a ready-to-train torch module.
        Output layer must produce raw logit (single neuron), BCEWithLogitsLoss
        is applied in the training loop.
        """

    def _create_sequences(self, features_scaled, target, dates):
        sequences, labels, target_indices, target_dates = [], [], [], []
        for target_index in range(self.lookback_days, len(features_scaled)):
            sequences.append(features_scaled[target_index - self.lookback_days:target_index])
            labels.append(target[target_index])
            target_indices.append(target_index)
            target_dates.append(dates.iloc[target_index] if dates is not None else None)
        return sequences, labels, target_indices, target_dates

    def _prepare_time_series_data(self, df: pd.DataFrame, feature_cols: list, target_col: str = "target") -> dict:
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        features = df[feature_cols].values
        target = df[target_col].values.astype(np.float32)  # binary 0/1
        dates = pd.to_datetime(df["date"]) if "date" in df.columns else None

        raw_count = len(df)
        train_end_raw = int(raw_count * self.train_ratio)
        val_end_raw = int(raw_count * (self.train_ratio + self.val_ratio))

        if train_end_raw <= self.lookback_days:
            raise ValueError(
                f"Train split too small for lookback={self.lookback_days}. "
                f"Need more than {self.lookback_days} rows."
            )
        if val_end_raw <= train_end_raw or val_end_raw >= raw_count:
            raise ValueError("Invalid train/val ratios for current dataset size.")

        # Fit scaler ONLY on train to prevent look-ahead bias
        self.feature_scaler.fit(features[:train_end_raw])
        features_scaled = self.feature_scaler.transform(features)

        sequences, labels, target_indices, target_dates = self._create_sequences(
            features_scaled, target, dates
        )

        partitions = {
            "train": {"X": [], "y": [], "target_indices": [], "target_dates": []},
            "val": {"X": [], "y": [], "target_indices": [], "target_dates": []},
            "test": {"X": [], "y": [], "target_indices": [], "target_dates": []},
        }

        for seq, lbl, tidx, tdate in zip(sequences, labels, target_indices, target_dates):
            if tidx < train_end_raw:
                part = "train"
            elif tidx < val_end_raw:
                part = "val"
            else:
                part = "test"
            partitions[part]["X"].append(seq)
            partitions[part]["y"].append(lbl)
            partitions[part]["target_indices"].append(tidx)
            partitions[part]["target_dates"].append(tdate)

        for part in partitions.values():
            part["X"] = np.asarray(part["X"], dtype=np.float32)
            part["y"] = np.asarray(part["y"], dtype=np.float32)
            part["target_dates"] = [
                v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else v
                for v in part["target_dates"]
            ]

        train_end_date = dates.iloc[train_end_raw - 1].strftime("%Y-%m-%d") if dates is not None else None
        val_end_date = dates.iloc[val_end_raw - 1].strftime("%Y-%m-%d") if dates is not None else None

        return {
            "train": partitions["train"],
            "val": partitions["val"],
            "test": partitions["test"],
            "train_end_date": train_end_date,
            "val_end_date": val_end_date,
            "lookback_days": self.lookback_days,
            "train_end_raw": train_end_raw,
            "val_end_raw": val_end_raw,
            "row_count": raw_count,
        }

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str = "target") -> dict:
        self.feature_cols = feature_cols
        logger.info(
            f"[{self.model_name}] Training (Classification) | "
            f"{len(feature_cols)} features | {len(df)} rows | Device: {self.device}"
        )

        split_data = self._prepare_time_series_data(df, feature_cols, target_col=target_col)
        train_split = split_data["train"]
        val_split = split_data["val"]
        test_split = split_data["test"]

        for split_name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
            if len(split["X"]) == 0:
                raise ValueError(f"Split '{split_name}' is empty.")

        logger.info(
            f"  Split (chronological): "
            f"train={len(train_split['X'])} | val={len(val_split['X'])} | test={len(test_split['X'])}"
        )

        # Check class balance in train set
        train_labels = train_split["y"]
        pos_ratio = train_labels.mean()
        logger.info(f"  Train class balance: {pos_ratio:.1%} UP / {1-pos_ratio:.1%} DOWN")

        self.model = self.build_model(input_size=len(feature_cols)).to(self.device)

        # pos_weight > 1 penalises false negatives (missed DOWNs)
        # pos_weight < 1 penalises false positives (wrong UPs)
        # We want fewer false UPs, so use weight < 1 when UP is majority
        neg_count = float((train_labels == 0).sum())
        pos_count = float((train_labels == 1).sum())
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(self.device)
        logger.info(f"  pos_weight (neg/pos): {pos_weight.item():.3f}")

        # BCEWithLogitsLoss = Sigmoid + BCE, numerically stable
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_split["X"]), torch.FloatTensor(train_split["y"])),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_split["X"]), torch.FloatTensor(val_split["y"])),
            batch_size=self.batch_size,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 20

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_losses.append(criterion(self.model(xb), yb).item())

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_losses))
            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)
            scheduler.step(avg_val)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch + 1}/{self.epochs} | "
                    f"Train: {avg_train:.4f} | Val: {avg_val:.4f}"
                )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        self._load_checkpoint()

        # ============================================================
        # Auto-optimise threshold trên validation set (thay vì cố định 0.5)
        # ============================================================
        self.threshold = self._find_optimal_threshold(val_split["X"], val_split["y"])
        logger.info(f"  Optimal threshold (val balanced accuracy): {self.threshold:.3f}")

        self.split_metadata = {
            "train_end_date": split_data["train_end_date"],
            "val_end_date": split_data["val_end_date"],
            "lookback_days": split_data["lookback_days"],
        }

        metrics = self._evaluate(test_split["X"], test_split["y"])
        metrics["val_loss"] = float(best_val_loss)
        metrics["epochs_trained"] = epoch + 1
        metrics["trained_at"] = datetime.now().isoformat()
        metrics["threshold"] = self.threshold
        metrics.update({
            "train_end_date": split_data["train_end_date"],
            "val_end_date": split_data["val_end_date"],
            "lookback_days": self.lookback_days,
            "feature_cols": list(self.feature_cols),
        })

        logger.info(
            f"[{self.model_name}] Acc: {metrics['accuracy']:.1f}% | "
            f"F1: {metrics['f1']:.4f} | "
            f"AUC: {metrics['roc_auc']:.4f}"
        )

        return metrics

    def _find_optimal_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Find validation threshold while avoiding one-class predictions."""
        self.model.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=self.batch_size,
        )

        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb.to(self.device)).cpu().numpy()
                all_logits.extend(logits)
                all_labels.extend(yb.numpy())

        probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
        labels = np.array(all_labels)

        best_score, best_f1, best_t = -1.0, -1.0, 0.5
        for t in np.arange(0.30, 0.71, 0.01):
            preds = (probs >= t).astype(int)
            score = balanced_accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, zero_division=0)
            if (score, f1) > (best_score, best_f1):
                best_score = float(score)
                best_f1 = f1
                best_t = float(t)

        return best_t

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Tính classification metrics trên test set."""
        self.model.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=self.batch_size,
        )

        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb.to(self.device)).cpu().numpy()
                all_logits.extend(logits)
                all_labels.extend(yb.numpy())

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)

        # Sigmoid để chuyển logit → xác suất
        probs = 1.0 / (1.0 + np.exp(-all_logits))
        preds = (probs >= self.threshold).astype(int)

        acc = accuracy_score(all_labels, preds) * 100
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, probs)
        except ValueError:
            auc = 0.5  # không tính được AUC nếu chỉ có 1 class

        return {
            "model_name": self.model_name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc),
            "test_loss": float(np.mean((probs - all_labels) ** 2)),
        }

    def predict_proba(self, df: pd.DataFrame) -> float:
        """
        Trả về xác suất P(giá ngày mai tăng) dựa trên lookback_days ngày gần nhất.
        """
        if len(df) < self.lookback_days:
            raise ValueError(
                f"Need at least {self.lookback_days} rows for prediction, got {len(df)}."
            )

        self.model.eval()
        features = df[self.feature_cols].values[-self.lookback_days:]
        features_scaled = self.feature_scaler.transform(features)

        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(X).cpu().item()

        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(prob)

    def predict_trend(self, df: pd.DataFrame) -> int:
        """Trả về nhãn xu hướng: 1 = Tăng, 0 = Giảm/Đi ngang."""
        return int(self.predict_proba(df) >= self.threshold)

    # Backward compatibility alias
    def predict_next(self, df: pd.DataFrame) -> float:
        """Alias cho predict_proba (backward compat)."""
        return self.predict_proba(df)

    def save(self, name=None):
        name = name or self.model_name
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"tcb_{name}.pt"

        # Serialize scaler — support both RobustScaler and MinMaxScaler
        scaler_state = {"scaler_type": type(self.feature_scaler).__name__}
        if hasattr(self.feature_scaler, "center_"):
            # RobustScaler
            scaler_state["center_"] = self.feature_scaler.center_.tolist()
            scaler_state["scale_"] = self.feature_scaler.scale_.tolist()
        elif hasattr(self.feature_scaler, "data_min_"):
            # MinMaxScaler (backward compat)
            for key in ["data_min_", "scale_", "data_range_", "data_max_"]:
                scaler_state[key] = getattr(self.feature_scaler, key).tolist()

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "feature_cols": self.feature_cols,
                "feature_scaler": scaler_state,
                "history": self.history,
                "saved_at": datetime.now().isoformat(),
                "lookback_days": self.lookback_days,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "threshold": self.threshold,
                "split_metadata": self.split_metadata,
                "task": "classification",
            },
            path,
        )
        logger.info(f"Model saved: {path}")

    def load(self, name=None):
        name = name or self.model_name
        path = MODEL_DIR / f"tcb_{name}.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.lookback_days = ckpt.get("lookback_days", self.lookback_days)
        self.epochs = ckpt.get("epochs", self.epochs)
        self.batch_size = ckpt.get("batch_size", self.batch_size)
        self.learning_rate = ckpt.get("learning_rate", self.learning_rate)
        self.train_ratio = ckpt.get("train_ratio", self.train_ratio)
        self.val_ratio = ckpt.get("val_ratio", self.val_ratio)
        self.threshold = ckpt.get("threshold", self.threshold)
        self.split_metadata = ckpt.get("split_metadata", {})

        self.feature_cols = ckpt["feature_cols"]
        self.model = self.build_model(input_size=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        scaler_data = ckpt.get("feature_scaler", {})
        scaler_type = scaler_data.get("scaler_type", "MinMaxScaler")

        if scaler_type == "RobustScaler" or "center_" in scaler_data:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            scaler.center_ = np.array(scaler_data["center_"])
            scaler.scale_ = np.array(scaler_data["scale_"])
            scaler.n_features_in_ = len(scaler_data["center_"])
        else:
            # Backward compat: load old MinMaxScaler format
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            for key in ["data_min_", "data_max_", "data_range_", "scale_"]:
                if key in scaler_data:
                    setattr(scaler, key, np.array(scaler_data[key]))
            if "data_min_" in scaler_data and "scale_" in scaler_data:
                scaler.min_ = -np.array(scaler_data["data_min_"]) * np.array(scaler_data["scale_"])
                scaler.n_features_in_ = len(scaler_data["data_min_"])

        self.feature_scaler = scaler
        logger.info(f"Model loaded: {path}")

    def _save_checkpoint(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_DIR / f"_checkpoint_{self.model_name}.pt")

    def _load_checkpoint(self):
        checkpoint_path = MODEL_DIR / f"_checkpoint_{self.model_name}.pt"
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
