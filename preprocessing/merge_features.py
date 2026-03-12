"""
Merge 3 bảng clean thành merged_features cho model.
═════════════════════════════════════════════════════
Phụ trách: Thành viên D
Branch: feature/merge-features
Chạy: python -m preprocessing.merge_features

Input:  clean_prices, clean_finance, clean_news
Output: merged_features table (sẵn sàng cho model)
"""
import pandas as pd
from loguru import logger
from database.connection import read_table, write_table

logger.add("logs/merge_features.log", rotation="1 week")


def merge_features():
    """
    Merge 3 bảng clean thành 1 bảng merged_features.

    Logic merge:
    - clean_prices: trục chính (mỗi ngày giao dịch = 1 row)
    - clean_finance: map theo quarter (mỗi ngày trong Q3 dùng số liệu Q3)
    - clean_news: aggregate theo ngày (mean sentiment, count)
    - target: giá close ngày hôm sau (shift -1)
    """
    logger.info("Merge features từ 3 nguồn...")

    # TODO: Thành viên D implement
    # Gợi ý:
    #
    # 1. Đọc 3 bảng clean:
    #    prices = read_table("clean_prices")
    #    finance = read_table("clean_finance")
    #    news = read_table("clean_news")
    #
    # 2. Merge finance vào prices theo quarter:
    #    - Từ date → xác định quarter ("2024-Q3")
    #    - LEFT JOIN clean_finance ON quarter
    #    - Forward fill nếu quarter chưa có data (dùng quý trước)
    #
    # 3. Aggregate news theo ngày:
    #    daily_news = news.groupby('date').agg(
    #        daily_sentiment=('sentiment_score', 'mean'),
    #        news_count=('sentiment_score', 'count')
    #    )
    #    - LEFT JOIN vào prices ON date
    #    - Fill 0 cho ngày không có tin
    #
    # 4. Tạo target column:
    #    df['target'] = df['close'].shift(-1)
    #    df = df.dropna(subset=['target'])
    #
    # 5. Drop rows có NaN (từ rolling windows của indicators)
    # 6. Lưu: write_table(df, "merged_features")

    raise NotImplementedError("Thành viên D cần implement hàm này")


if __name__ == "__main__":
    merge_features()
