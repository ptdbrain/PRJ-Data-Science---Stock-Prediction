"""Load saved model and predict next-day close using last LOOKBACK_DAYS rows."""
import argparse
from pathlib import Path
import sqlite3
import pandas as pd
import logging

from config import settings
from models.lstm_model import LSTMPredictor


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_merged_features(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['date'])
    finally:
        conn.close()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=str(settings.DB_PATH))
    parser.add_argument('--table', default=settings.TABLE_MERGED_FEATURES)
    parser.add_argument('--name', default='lstm_tcb')
    args = parser.parse_args()

    df = load_merged_features(Path(args.db), args.table)
    df = df.sort_values('date').reset_index(drop=True) if 'date' in df.columns else df

    model = LSTMPredictor()
    model.load(name=args.name)

    # Ensure feature_cols exist in dataframe
    for c in model.feature_cols:
        if c not in df.columns:
            raise RuntimeError(f"Required feature missing in data: {c}")

    # Use last LOOKBACK_DAYS rows
    recent = df.tail(settings.LOOKBACK_DAYS)
    pred = model.predict_next(recent)
    logger.info(f"Predicted next close: {pred:,.2f}")


if __name__ == '__main__':
    main()
