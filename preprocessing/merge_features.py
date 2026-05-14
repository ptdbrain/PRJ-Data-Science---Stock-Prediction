"""
Merge cleaned price / finance / news features into merged_features for models.
"""
import pandas as pd
from utils.logger import logger

from config.settings import (
    ALL_FEATURES,
    TABLE_CLEAN_NEWS,
    TABLE_CLEAN_PRICES,
    TABLE_FEATURES_FINANCE,
    TABLE_MERGED_FEATURES,
)
from database.connection import read_table, write_table


logger.add("logs/merge_features.log", rotation="1 week")


def _load_clean_prices() -> pd.DataFrame:
    prices = read_table(TABLE_CLEAN_PRICES)
    if prices.empty:
        raise ValueError("clean_prices rỗng — không thể merge")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    prices["trade_date"] = pd.to_datetime(prices["date"])
    return prices.sort_values("trade_date").reset_index(drop=True)


def _merge_finance_features(prices: pd.DataFrame) -> pd.DataFrame:
    try:
        finance = read_table(TABLE_FEATURES_FINANCE)
    except Exception:
        logger.warning("Không tìm thấy features_finance — tiếp tục không có finance features")
        return prices.copy()

    if finance.empty:
        logger.warning("features_finance rỗng — tiếp tục không có finance features")
        return prices.copy()

    finance = finance.copy()
    finance["effective_date"] = pd.to_datetime(finance["effective_date"])
    finance["period_end_date"] = pd.to_datetime(finance["period_end_date"]).dt.strftime("%Y-%m-%d")
    finance = finance.rename(columns={"date": "finance_quarter"})
    finance = finance.sort_values(["effective_date", "finance_quarter"]).reset_index(drop=True)

    finance_columns = [
        column
        for column in finance.columns
        if column not in {"id", "symbol", "created_at"}
    ]
    finance = finance[finance_columns].copy()
    finance["effective_date_dt"] = finance["effective_date"]

    merged = pd.merge_asof(
        prices.sort_values("trade_date"),
        finance.sort_values("effective_date_dt"),
        left_on="trade_date",
        right_on="effective_date_dt",
        direction="backward",
    )

    merged["effective_date"] = pd.to_datetime(merged["effective_date"]).dt.strftime("%Y-%m-%d")
    merged = merged.drop(columns=["effective_date_dt"], errors="ignore")
    return merged


def _merge_news_features(merged: pd.DataFrame) -> pd.DataFrame:
    try:
        news = read_table(TABLE_CLEAN_NEWS)
    except Exception:
        logger.warning("Không tìm thấy clean_news — dùng giá trị news mặc định")
        news = pd.DataFrame()

    if news.empty:
        merged["daily_sentiment"] = 0.0
        merged["news_count"] = 0
        merged["embedding_score_mean"] = 0.0
        merged["embedding_score_std"] = 0.0
        merged["tfidf_sentiment"] = 0.0
        return merged

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.strftime("%Y-%m-%d")

    daily_news = (
        news.groupby("date", sort=False)
        .agg(
            daily_sentiment=("sentiment_score", "mean"),
            news_count=("date", "size"),
            embedding_score_mean=("embedding_score", "mean"),
            embedding_score_std=("embedding_score", "std"),
            tfidf_sentiment=("tfidf_sentiment", "mean"),   # TF-IDF classical NLP
        )
        .reset_index()
    )

    merged = merged.merge(daily_news, on="date", how="left")
    merged["daily_sentiment"] = merged["daily_sentiment"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["embedding_score_mean"] = merged["embedding_score_mean"].fillna(0.0)
    merged["embedding_score_std"] = merged["embedding_score_std"].fillna(0.0)
    merged["tfidf_sentiment"] = merged["tfidf_sentiment"].fillna(0.0)
    return merged


def merge_features():
    logger.info("Merge features từ 3 nguồn...")

    prices = _load_clean_prices()
    merged = _merge_finance_features(prices)
    merged = _merge_news_features(merged)

    # ================================================================
    # Target: nhãn phân loại nhị phân (Classification)
    # 1 = giá ngày mai TĂNG so với hôm nay
    # 0 = giá ngày mai GIẢM hoặc ĐI NGANG
    # Lưu ý: shift(-1) → dùng giá ngày khác (không có look-ahead bias
    # vì cột này chỉ xuất hiện lúc train, không dùng khi inference)
    # ================================================================
    # Lưu giá thực tế của ngày mai để hiển thị trên dashboard.
    # Dòng cuối chưa có giá ngày mai nên không được ép thành target=0.
    merged["next_close"] = merged["close"].shift(-1)
    merged = merged.dropna(subset=["next_close"]).reset_index(drop=True)
    merged["target"] = (merged["next_close"] > merged["close"]).astype(int)

    # Các cột optional (có thể thiếu hoặc toàn NaN với data cũ) — không ép dropna
    OPTIONAL_FEATURES = {"eps", "eps_yoy", "tfidf_sentiment", "daily_tfidf_sentiment",
                         "embedding_score_mean", "embedding_score_std",
                         "obv_change", "volatility_pct"}
    required_feature_columns = [
        column for column in ALL_FEATURES
        if column in merged.columns and column not in OPTIONAL_FEATURES
    ]
    merged = merged.dropna(subset=required_feature_columns + ["target"]).reset_index(drop=True)

    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged = merged.drop(columns=["trade_date"], errors="ignore")

    write_table(merged, TABLE_MERGED_FEATURES, if_exists="replace")
    logger.info(f"Saved merged_features ({len(merged)} rows, {merged.shape[1]} cols)")
    return merged


if __name__ == "__main__":
    merge_features()
