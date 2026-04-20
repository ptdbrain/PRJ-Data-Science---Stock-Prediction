"""
Chạy trọn pipeline: schema → giá → tài chính → (tin nếu có) → merge → train LSTM.

Cách chạy từ thư mục gốc repo:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-news
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full stock LSTM pipeline")
    parser.add_argument("--skip-news", action="store_true", help="Không chạy process_news")
    parser.add_argument("--skip-train", action="store_true", help="Chỉ dữ liệu, không train LSTM")
    args = parser.parse_args()

    from database.schema import create_all_tables
    from database.connection import table_exists, table_row_count

    logger.info("Bước 1: Tạo bảng SQLite (schema)")
    create_all_tables()

    logger.info("Bước 2: Thu thập & làm sạch giá OHLCV")
    from data_collection.collect_prices import collect_prices
    from preprocessing.process_prices import process_prices

    collect_prices()
    process_prices()

    logger.info("Bước 3: Thu thập & xử lý chỉ số tài chính (ratio)")
    from data_collection.collect_finance import crawl_ratio_api
    from preprocessing.process_finance import process_finance

    fin = crawl_ratio_api()
    if fin is None:
        logger.warning("collect_finance không trả về DataFrame — kiểm tra vnstock / mạng.")
    process_finance()

    if not args.skip_news and table_exists("raw_news") and table_row_count("raw_news") > 0:
        logger.info("Bước 4: Tiền xử lý tin tức + sentiment")
        from preprocessing.process_news import process_news

        process_news()
    else:
        logger.info("Bước 4: Bỏ qua tin (không có raw_news hoặc --skip-news)")

    logger.info("Bước 5: Merge đặc trưng → merged_features")
    from preprocessing.merge_features import merge_features

    merged = merge_features()
    if merged is None or merged.empty:
        raise SystemExit("merge_features thất bại hoặc rỗng — dừng pipeline.")

    if args.skip_train:
        logger.info("Đã --skip-train: kết thúc sau merge.")
        return

    logger.info("Bước 6: Train LSTM")
    train_script = ROOT / "scripts" / "train_lstm.py"
    r = subprocess.run([sys.executable, str(train_script)], cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"train_lstm.py thoát với mã {r.returncode}")


if __name__ == "__main__":
    main()
