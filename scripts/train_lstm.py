"""Train LSTM on `merged_features` and save metrics + model."""
import argparse
import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

try:
    from config import settings
except Exception:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from config import settings

from models.lstm_model import LSTMPredictor


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(settings.DB_PATH))
    parser.add_argument("--table", default=settings.TABLE_MERGED_FEATURES)
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE)
    parser.add_argument("--name", default=settings.DEFAULT_MODEL_NAME)
    parser.add_argument("--save-dir", default=str(settings.MODEL_DIR))
    return parser


def load_merged_features(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=["date"])
    finally:
        conn.close()
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    if "target" not in df.columns:
        if "close" not in df.columns:
            raise ValueError("`close` column required to build target")
        df["next_close"] = df["close"].shift(-1)
        df = df.dropna(subset=["next_close"]).reset_index(drop=True)
        df["target"] = (df["next_close"] > df["close"]).astype(int)
    else:
        target_values = df["target"].dropna().unique()
        if not set(target_values).issubset({0, 1}):
            raise ValueError(
                "`target` must be binary 0/1 for classification. "
                "Run preprocessing.merge_features again to rebuild labels."
            )

    return df


def pick_features(df: pd.DataFrame):
    feature_cols = [column for column in settings.ALL_FEATURES if column in df.columns]
    if not feature_cols:
        raise ValueError("No features from settings.ALL_FEATURES found in input df")
    return feature_cols


def save_metrics(metrics: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{metrics.get('model_name', 'model')}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved: {path}")


def main():
    args = build_parser().parse_args()

    logger.info("Loading merged features from DB...")
    df = load_merged_features(Path(args.db), args.table)
    df = prepare_df(df)
    feature_cols = pick_features(df)
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            "Không còn dòng sau khi dropna theo feature + target "
            "— kiểm tra merged_features và ALL_FEATURES."
        )

    logger.info(f"Training rows: {len(df)} | features: {len(feature_cols)}")
    logger.info(f"Hyperparams: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    model = LSTMPredictor(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    metrics = model.fit(df, feature_cols, target_col="target")
    metrics["train_args"] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "feature_count": len(feature_cols),
    }

    model.save(name=args.name)
    save_metrics(metrics, Path(args.save_dir))

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
