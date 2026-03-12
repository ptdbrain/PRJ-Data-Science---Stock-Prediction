"""
Base class cho tất cả prediction models.
══════════════════════════════════════
Mỗi model (LSTM, GRU, Transformer) kế thừa class này.
Đảm bảo mọi model có cùng interface → dễ so sánh.

Phụ trách: Thành viên D (viết base), E (review)
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from datetime import datetime

from config.settings import (
    DEVICE, LOOKBACK_DAYS, EPOCHS, BATCH_SIZE,
    LEARNING_RATE, TRAIN_RATIO, VAL_RATIO, MODEL_DIR
)


class BasePredictor(ABC):
    """
    Base class — mọi model phải kế thừa.

    Cách dùng:
        model = LSTMPredictor()
        metrics = model.fit(df, feature_cols)    # Train
        price = model.predict_next(df)           # Predict ngày mai
        model.save()                             # Lưu
        model.load()                             # Load
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: nn.Module = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_cols = []
        self.history = {'train_loss': [], 'val_loss': []}

    @abstractmethod
    def build_model(self, input_size: int) -> nn.Module:
        """Subclass PHẢI implement: trả về nn.Module."""
        pass

    def create_sequences(self, features, target, lookback=LOOKBACK_DAYS):
        """Tạo sliding window sequences cho time series."""
        X, y = [], []
        for i in range(lookback, len(features)):
            X.append(features[i - lookback:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame, feature_cols: list,
            target_col: str = 'target') -> dict:
        """
        Train model.

        Args:
            df: DataFrame chứa features + target
            feature_cols: list tên cột features
            target_col: tên cột target (giá ngày hôm sau)

        Returns:
            dict metrics: rmse, mae, mape, directional_accuracy, ...
        """
        self.feature_cols = feature_cols
        logger.info(f"[{self.model_name}] Training | "
                     f"{len(feature_cols)} features | {len(df)} rows | Device: {DEVICE}")

        # --- Scale ---
        features = df[feature_cols].values
        target = df[target_col].values
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

        # --- Sequences ---
        X, y = self.create_sequences(features_scaled, target_scaled)

        # --- Split (THEO THỜI GIAN, không random) ---
        n = len(X)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"  Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")

        # --- Build model ---
        self.model = self.build_model(input_size=len(feature_cols)).to(DEVICE)

        # --- Training loop ---
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=BATCH_SIZE
        )

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20

        for epoch in range(EPOCHS):
            # Train
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_losses.append(criterion(self.model(xb), yb).item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            self.history['train_loss'].append(avg_train)
            self.history['val_loss'].append(avg_val)
            scheduler.step(avg_val)

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{EPOCHS} | "
                             f"Train: {avg_train:.6f} | Val: {avg_val:.6f}")

            # Early stopping + save best
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best checkpoint
        self._load_checkpoint()

        # Evaluate on test set
        metrics = self._evaluate(X_test, y_test)
        metrics['val_loss'] = float(best_val_loss)
        metrics['epochs_trained'] = epoch + 1
        metrics['trained_at'] = datetime.now().isoformat()

        logger.info(f"[{self.model_name}] ✅ RMSE: {metrics['rmse']:,.0f} VND | "
                     f"MAPE: {metrics['mape']:.2f}% | "
                     f"Direction: {metrics['directional_accuracy']:.1f}%")

        return metrics

    def _evaluate(self, X_test, y_test) -> dict:
        """Đánh giá model trên test set."""
        self.model.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=BATCH_SIZE
        )

        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in loader:
                all_preds.extend(self.model(xb.to(DEVICE)).cpu().numpy())
                all_targets.extend(yb.numpy())

        # Inverse transform → giá thật (VND)
        preds = self.target_scaler.inverse_transform(
            np.array(all_preds).reshape(-1, 1)).flatten()
        targets = self.target_scaler.inverse_transform(
            np.array(all_targets).reshape(-1, 1)).flatten()

        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mae = np.mean(np.abs(preds - targets))
        mape = np.mean(np.abs((targets - preds) / targets)) * 100

        # Directional accuracy: predict đúng hướng tăng/giảm
        dir_correct = np.sum(
            np.sign(preds[1:] - preds[:-1]) ==
            np.sign(targets[1:] - targets[:-1])
        )
        dir_acc = dir_correct / max(len(preds) - 1, 1) * 100

        return {
            'model_name': self.model_name,
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'directional_accuracy': float(dir_acc),
            'test_loss': float(np.mean((np.array(all_preds) - np.array(all_targets))**2)),
        }

    def predict_next(self, df: pd.DataFrame) -> float:
        """
        Predict giá ngày tiếp theo.

        Args:
            df: DataFrame chứa ít nhất LOOKBACK_DAYS dòng gần nhất

        Returns:
            Giá dự đoán (VND)
        """
        self.model.eval()
        features = df[self.feature_cols].values[-LOOKBACK_DAYS:]
        features_scaled = self.feature_scaler.transform(features)

        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_scaled = self.model(X).cpu().numpy()

        return float(self.target_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)).flatten()[0])

    def save(self, name=None):
        """Lưu model + scalers."""
        name = name or self.model_name
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"tcb_{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'feature_cols': self.feature_cols,
            'feature_scaler': {k: getattr(self.feature_scaler, k).tolist()
                               for k in ['data_min_', 'scale_', 'data_range_', 'data_max_']},
            'target_scaler': {k: getattr(self.target_scaler, k).tolist()
                              for k in ['data_min_', 'scale_', 'data_range_', 'data_max_']},
            'history': self.history,
            'saved_at': datetime.now().isoformat()
        }, path)
        logger.info(f"💾 Model saved: {path}")

    def load(self, name=None):
        """Load model đã lưu."""
        name = name or self.model_name
        path = MODEL_DIR / f"tcb_{name}.pt"
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

        self.feature_cols = ckpt['feature_cols']
        self.model = self.build_model(input_size=len(self.feature_cols)).to(DEVICE)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        # Restore scalers
        for attr, key in [('feature_scaler', 'feature_scaler'),
                          ('target_scaler', 'target_scaler')]:
            scaler = MinMaxScaler()
            for k, v in ckpt[key].items():
                setattr(scaler, k, np.array(v))
            scaler.n_features_in_ = len(ckpt[key]['data_min_'])
            setattr(self, attr, scaler)

        logger.info(f"📂 Model loaded: {path}")

    def _save_checkpoint(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_DIR / f"_checkpoint_{self.model_name}.pt")

    def _load_checkpoint(self):
        self.model.load_state_dict(
            torch.load(MODEL_DIR / f"_checkpoint_{self.model_name}.pt",
                        map_location=DEVICE, weights_only=True))
