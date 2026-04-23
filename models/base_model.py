"""
Base class cho tất cả prediction models.
"""
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.logger import logger
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config.settings import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    LOOKBACK_DAYS,
    MODEL_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
)


class BasePredictor(ABC):
    """Base predictor shared by LSTM/GRU/Transformer models."""

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
    ):
        self.model_name = model_name
        self.lookback_days = lookback_days
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.device = device

        self.model: nn.Module | None = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_cols: list[str] = []
        self.history = {"train_loss": [], "val_loss": []}
        self.split_metadata: dict[str, object] = {}

    @abstractmethod
    def build_model(self, input_size: int) -> nn.Module:
        """Subclass must return a ready-to-train torch module."""

    def _create_sequences(self, features_scaled, target_scaled, dates):
        sequences = []
        labels = []
        target_indices = []
        target_dates = []

        for target_index in range(self.lookback_days, len(features_scaled)):
            sequences.append(features_scaled[target_index - self.lookback_days:target_index])
            labels.append(target_scaled[target_index])
            target_indices.append(target_index)
            target_dates.append(dates.iloc[target_index] if dates is not None else None)

        return sequences, labels, target_indices, target_dates

    def _prepare_time_series_data(self, df: pd.DataFrame, feature_cols: list, target_col: str = "target") -> dict:
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        features = df[feature_cols].values
        target = df[target_col].values
        dates = pd.to_datetime(df["date"]) if "date" in df.columns else None

        raw_count = len(df)
        train_end_raw = int(raw_count * self.train_ratio)
        val_end_raw = int(raw_count * (self.train_ratio + self.val_ratio))

        if train_end_raw <= self.lookback_days:
            raise ValueError(
                f"Train split too small for lookback={self.lookback_days}. "
                f"Need more than {self.lookback_days} rows in train split."
            )
        if val_end_raw <= train_end_raw or val_end_raw >= raw_count:
            raise ValueError("Invalid train/val ratios for current dataset size.")

        self.feature_scaler.fit(features[:train_end_raw])
        self.target_scaler.fit(target[:train_end_raw].reshape(-1, 1))

        features_scaled = self.feature_scaler.transform(features)
        target_scaled = self.target_scaler.transform(target.reshape(-1, 1)).flatten()

        sequences, labels, target_indices, target_dates = self._create_sequences(
            features_scaled,
            target_scaled,
            dates,
        )

        partitions = {
            "train": {"X": [], "y": [], "target_indices": [], "target_dates": []},
            "val": {"X": [], "y": [], "target_indices": [], "target_dates": []},
            "test": {"X": [], "y": [], "target_indices": [], "target_dates": []},
        }

        for sequence, label, target_index, target_date in zip(sequences, labels, target_indices, target_dates):
            if target_index < train_end_raw:
                partition = "train"
            elif target_index < val_end_raw:
                partition = "val"
            else:
                partition = "test"

            partitions[partition]["X"].append(sequence)
            partitions[partition]["y"].append(label)
            partitions[partition]["target_indices"].append(target_index)
            partitions[partition]["target_dates"].append(target_date)

        for partition in partitions.values():
            partition["X"] = np.asarray(partition["X"], dtype=np.float32)
            partition["y"] = np.asarray(partition["y"], dtype=np.float32)
            partition["target_dates"] = [
                value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else value
                for value in partition["target_dates"]
            ]

        train_end_date = dates.iloc[train_end_raw - 1].strftime("%Y-%m-%d") if dates is not None else None
        val_end_date = dates.iloc[val_end_raw - 1].strftime("%Y-%m-%d") if dates is not None else None

        return {
            "train": partitions["train"],
            "val": partitions["val"],
            "test": partitions["test"],
            "train_target_indices": partitions["train"]["target_indices"],
            "val_target_indices": partitions["val"]["target_indices"],
            "test_target_indices": partitions["test"]["target_indices"],
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
            f"[{self.model_name}] Training | "
            f"{len(feature_cols)} features | {len(df)} rows | Device: {self.device}"
        )

        split_data = self._prepare_time_series_data(df, feature_cols, target_col=target_col)
        train_split = split_data["train"]
        val_split = split_data["val"]
        test_split = split_data["test"]

        if len(train_split["X"]) == 0 or len(val_split["X"]) == 0 or len(test_split["X"]) == 0:
            raise ValueError("Train/val/test partitions must all contain at least one sequence.")

        logger.info(
            "  Split by target index: "
            f"train={len(train_split['X'])} | val={len(val_split['X'])} | test={len(test_split['X'])}"
        )

        self.model = self.build_model(input_size=len(feature_cols)).to(self.device)

        criterion = nn.MSELoss()
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
                loss = criterion(self.model(xb), yb)
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
                    f"Train: {avg_train:.6f} | Val: {avg_val:.6f}"
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

        self.split_metadata = {
            "train_end_date": split_data["train_end_date"],
            "val_end_date": split_data["val_end_date"],
            "lookback_days": split_data["lookback_days"],
            "train_target_indices": split_data["train_target_indices"],
            "val_target_indices": split_data["val_target_indices"],
            "test_target_indices": split_data["test_target_indices"],
        }

        metrics = self._evaluate(test_split["X"], test_split["y"])
        metrics["val_loss"] = float(best_val_loss)
        metrics["epochs_trained"] = epoch + 1
        metrics["trained_at"] = datetime.now().isoformat()
        metrics.update(
            {
                "train_end_date": split_data["train_end_date"],
                "val_end_date": split_data["val_end_date"],
                "lookback_days": self.lookback_days,
                "feature_cols": list(self.feature_cols),
            }
        )

        logger.info(
            f"[{self.model_name}] RMSE: {metrics['rmse']:,.0f} VND | "
            f"MAPE: {metrics['mape']:.2f}% | "
            f"Direction: {metrics['directional_accuracy']:.1f}%"
        )

        return metrics

    def _evaluate(self, X_test, y_test) -> dict:
        self.model.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=self.batch_size,
        )

        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in loader:
                all_preds.extend(self.model(xb.to(self.device)).cpu().numpy())
                all_targets.extend(yb.numpy())

        preds = self.target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
        targets = self.target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mae = np.mean(np.abs(preds - targets))
        mape = np.mean(np.abs((targets - preds) / np.where(targets == 0, 1, targets))) * 100

        if len(preds) > 1:
            pred_direction = np.sign(preds[1:] - targets[:-1])
            actual_direction = np.sign(targets[1:] - targets[:-1])
            dir_correct = np.sum(pred_direction == actual_direction)
            dir_acc = dir_correct / (len(preds) - 1) * 100
        else:
            dir_acc = 0.0

        return {
            "model_name": self.model_name,
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "directional_accuracy": float(dir_acc),
            "test_loss": float(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)),
        }

    def predict_next(self, df: pd.DataFrame) -> float:
        if len(df) < self.lookback_days:
            raise ValueError(
                f"Need at least {self.lookback_days} rows for prediction, got {len(df)}."
            )

        self.model.eval()
        features = df[self.feature_cols].values[-self.lookback_days:]
        features_scaled = self.feature_scaler.transform(features)

        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(X).cpu().numpy()

        return float(self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0])

    def save(self, name=None):
        name = name or self.model_name
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"tcb_{name}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "feature_cols": self.feature_cols,
                "feature_scaler": {
                    key: getattr(self.feature_scaler, key).tolist()
                    for key in ["data_min_", "scale_", "data_range_", "data_max_"]
                },
                "target_scaler": {
                    key: getattr(self.target_scaler, key).tolist()
                    for key in ["data_min_", "scale_", "data_range_", "data_max_"]
                },
                "history": self.history,
                "saved_at": datetime.now().isoformat(),
                "lookback_days": self.lookback_days,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "split_metadata": self.split_metadata,
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
        self.split_metadata = ckpt.get("split_metadata", {})

        self.feature_cols = ckpt["feature_cols"]
        self.model = self.build_model(input_size=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        for attr, key in [("feature_scaler", "feature_scaler"), ("target_scaler", "target_scaler")]:
            scaler = MinMaxScaler()
            for scaler_key, value in ckpt[key].items():
                setattr(scaler, scaler_key, np.array(value))
            scaler.n_features_in_ = len(ckpt[key]["data_min_"])
            setattr(self, attr, scaler)

        logger.info(f"Model loaded: {path}")

    def _save_checkpoint(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_DIR / f"_checkpoint_{self.model_name}.pt")

    def _load_checkpoint(self):
        checkpoint_path = MODEL_DIR / f"_checkpoint_{self.model_name}.pt"
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
