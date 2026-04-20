"""
Merge 3 bảng clean thành merged_features cho model.
═════════════════════════════════════════════════════
Chạy: python -m preprocessing.merge_features

Input:  clean_prices, clean_finance, clean_news
Output: merged_features table (sẵn sàng cho model)
"""
import re
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

    # 1. Read clean_prices
    try:
        prices = read_table("clean_prices")
    except Exception as e:
        logger.error(f"Không thể đọc clean_prices: {e}")
        return None

    if prices.empty:
        logger.error("clean_prices rỗng — không thể merge")
        return None

    # Normalize date column
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date']).dt.strftime('%Y-%m-%d')
    prices = prices.sort_values('date').reset_index(drop=True)

    # 2. Read finance features (try multiple possible table names)
    finance = None
    for fname in ('features_finance', 'clean_finance', 'raw_finance'):
        try:
            df_fin = read_table(fname)
            if not df_fin.empty:
                finance = df_fin.copy()
                finance_table = fname
                logger.info(f"Loaded finance from table: {fname} ({len(finance)} rows)")
                break
        except Exception:
            continue

    # Prepare quarter column on prices
    prices_dt = pd.to_datetime(prices['date'])
    prices['quarter'] = prices_dt.dt.year.astype(str) + '-Q' + prices_dt.dt.quarter.astype(str)

    merged = prices.copy()

    if finance is not None:
        # Convert finance date -> quarter (support both 'YYYY-Qx' and real dates)
        def _to_quarter(v):
            try:
                if isinstance(v, str) and '-Q' in v:
                    return v
                # vnstock / bản cũ: "2024Q3"
                m = re.match(r"^(\d{4})\s*Q\s*([1-4])$", str(v).strip(), re.I)
                if m:
                    return f"{m.group(1)}-Q{m.group(2)}"
                dt = pd.to_datetime(v)
                q = ((dt.month - 1) // 3) + 1
                return f"{dt.year}-Q{q}"
            except Exception:
                return None

        finance = finance.copy()
        if 'date' in finance.columns:
            finance['quarter'] = finance['date'].apply(_to_quarter)
        elif 'quarter' not in finance.columns:
            finance['quarter'] = None

        # Select numeric feature columns from finance
        finance_exclude = {'id', 'symbol', 'date', 'quarter', 'created_at'}
        finance_cols = [c for c in finance.columns if c not in finance_exclude]

        # Keep only quarter + features
        finance_q = finance[['quarter'] + finance_cols].drop_duplicates(subset=['quarter'], keep='last')

        # Merge into prices by quarter
        merged = merged.merge(finance_q, on='quarter', how='left')

        # Ensure numeric types and forward-fill finance features
        for c in finance_cols:
            try:
                merged[c] = pd.to_numeric(merged[c], errors='coerce')
            except Exception:
                pass
        merged = merged.sort_values('date').reset_index(drop=True)
        if finance_cols:
            merged[finance_cols] = merged[finance_cols].ffill()
    else:
        logger.warning("Không tìm thấy bảng finance — bỏ qua merge finance")

    # 3. Read and aggregate news
    news = None
    try:
        news = read_table('clean_news')
    except Exception:
        logger.warning('Không tìm thấy clean_news — bỏ qua merge news')

    if news is not None and not news.empty:
        news = news.copy()
        # Normalize date format
        news['date'] = pd.to_datetime(news['date']).dt.strftime('%Y-%m-%d')

        daily_records = []
        for dt, g in news.groupby('date', sort=False):
            rec = {'date': dt, 'news_count': int(len(g))}
            # sentiment
            if 'sentiment_score' in g.columns:
                rec['daily_sentiment'] = float(g['sentiment_score'].astype(float).mean())
            else:
                rec['daily_sentiment'] = None
            # embedding_score
            if 'embedding_score' in g.columns:
                rec['embedding_score_mean'] = float(g['embedding_score'].astype(float).mean())
                rec['embedding_score_std'] = float(g['embedding_score'].astype(float).std()) if len(g) > 1 else 0.0
            daily_records.append(rec)

        daily_news = pd.DataFrame(daily_records)

        # Merge into merged by date
        merged = merged.merge(daily_news, on='date', how='left')

        # Fill news missing with neutral defaults
        if 'daily_sentiment' in merged.columns:
            merged['daily_sentiment'] = merged['daily_sentiment'].fillna(0.0)
        if 'news_count' in merged.columns:
            merged['news_count'] = merged['news_count'].fillna(0).astype(int)
        if 'embedding_score_mean' in merged.columns:
            merged['embedding_score_mean'] = merged['embedding_score_mean'].fillna(0.0)
        if 'embedding_score_std' in merged.columns:
            merged['embedding_score_std'] = merged['embedding_score_std'].fillna(0.0)
    else:
        logger.info('No news data to merge; adding default news columns')
        merged['daily_sentiment'] = 0.0
        merged['news_count'] = 0

    # 4. Create target = next-day close
    merged['target'] = merged['close'].shift(-1)

    # 5. Drop rows with NaN target and any remaining NaNs
    before = len(merged)
    merged = merged.dropna(subset=['target']).reset_index(drop=True)
    # Optionally drop rows with any NaN in feature columns
    merged = merged.dropna().reset_index(drop=True)
    after = len(merged)
    logger.info(f"Merged features: rows {before} -> {after} after dropping NA and target")

    # 6. Save merged_features
    write_table(merged, 'merged_features')
    logger.info(f"Saved merged_features ({len(merged)} rows, {merged.shape[1]} cols)")
    return merged


if __name__ == "__main__":
    merge_features()
