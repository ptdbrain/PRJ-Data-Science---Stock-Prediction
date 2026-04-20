"""Train LSTM on `merged_features` and save metrics + model.

Usage:
    python scripts/train_lstm.py --epochs 50 --batch-size 64
"""
import argparse
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

from config import settings

from models.lstm_model import LSTMPredictor
import models.base_model as base_model


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_merged_features(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['date'])
    finally:
        conn.close()
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sắp xếp theo thời gian và tạo target; dropna theo cột feature thực tế gọi sau pick_features()."""
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    if 'target' not in df.columns:
        if 'close' not in df.columns:
            raise ValueError('`close` column required to build target')
        df['target'] = df['close'].shift(-1)

    return df


def pick_features(df: pd.DataFrame):
    # choose features that are present in dataframe
    feature_cols = [c for c in settings.ALL_FEATURES if c in df.columns]
    if not feature_cols:
        raise ValueError('No features from settings.ALL_FEATURES found in input df')
    return feature_cols


def save_metrics(metrics: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{metrics.get('model_name','model')}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=str(settings.DB_PATH))
    parser.add_argument('--table', default=settings.TABLE_MERGED_FEATURES)
    parser.add_argument('--epochs', type=int, default=settings.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=settings.LEARNING_RATE)
    parser.add_argument('--name', default='lstm_tcb')
    parser.add_argument('--save-dir', default=str(settings.MODEL_DIR))
    args = parser.parse_args()

    logger.info('Loading merged features from DB...')
    df = load_merged_features(Path(args.db), args.table)
    df = prepare_df(df)
    feature_cols = pick_features(df)
    df = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    if df.empty:
        raise RuntimeError('Không còn dòng sau khi dropna theo feature + target — kiểm tra merged_features và ALL_FEATURES.')

    # Monkeypatch base_model training hyperparams (so BasePredictor.fit uses our args)
    base_model.EPOCHS = args.epochs
    base_model.BATCH_SIZE = args.batch_size
    base_model.LEARNING_RATE = args.lr

    logger.info(f"Training rows: {len(df)} | features: {len(feature_cols)}")

    model = LSTMPredictor()
    metrics = model.fit(df, feature_cols, target_col='target')
    metrics['train_args'] = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'feature_count': len(feature_cols)
    }

    # Save model and metrics
    model.save(name=args.name)
    save_metrics(metrics, Path(args.save_dir))

    logger.info('Training finished.')


if __name__ == '__main__':
    main()
