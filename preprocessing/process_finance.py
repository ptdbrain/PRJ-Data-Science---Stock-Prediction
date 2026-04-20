"""
Process financial data: Clean + Feature Engineering
═════════════════════════════════════════════════════

STAGE 2 + STAGE 3 Combined:
  ✓ Load full raw data from database (raw_finance table)
    ✓ Select ROE, ROA, debt_to_equity, net_profit_margin, financial_leverage
  ✓ Handle zero values (data quality)
  ✓ Forward fill missing quarters
  ✓ Calculate YoY change (4-quarter lag)
  ✓ Save features to database (không normalize — để base_model thống nhất scale)

NOTE: StandardScaler đã được bỏ khỏi bước này để tránh double-scaling.
      Việc normalize toàn bộ features (giá + tài chính + sentiment) được
      thực hiện tập trung trong BasePredictor.fit(), chỉ fit trên train set.

Chạy: python -m preprocessing.process_finance
Input:  database raw_finance table
Output: database features_finance table
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_connection
from database.schema import recreate_features_finance_table
from config.settings import SYMBOL

logger.add("logs/process_finance.log", rotation="1 week")

TARGET_QUARTERS = 24


def load_raw_from_database():
    """Đọc dữ liệu gốc từ database (raw_finance table)"""
    try:
        conn = get_connection()
        query = "SELECT * FROM raw_finance WHERE symbol = ? ORDER BY date"
        df = pd.read_sql_query(query, conn, params=(SYMBOL,))
        conn.close()

        if df.empty:
            logger.error("Không có dữ liệu trong raw_finance!")
            return None

        logger.info(f"Loaded {len(df)} rows từ raw_finance")
        return df

    except Exception as e:
        logger.error(f"Lỗi khi đọc raw_finance: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_features_to_database(df):
    """Lưu dữ liệu đã xử lý vào database (features_finance table)"""
    try:
        # Recreate table để đảm bảo schema mới nhất
        recreate_features_finance_table()

        conn = get_connection()

        # Clear bảng cũ (cùng symbol)
        conn.execute("DELETE FROM features_finance WHERE symbol = ?", (SYMBOL,))

        # Insert từng row
        for _, row in df.iterrows():
            values = {
                'symbol': row.get('symbol', SYMBOL),
                'date': str(row.get('date')),
                'roe': row.get('roe'),
                'roa': row.get('roa'),
                'debt_to_equity': row.get('debt_to_equity'),
                'net_profit_margin': row.get('net_profit_margin'),
                'financial_leverage': row.get('financial_leverage'),
                'roe_yoy': row.get('roe_yoy'),
                'roa_yoy': row.get('roa_yoy'),
                'roe_lag4': row.get('roe_lag4'),
                'roa_lag4': row.get('roa_lag4'),
            }

            def _to_sql_param(key, val):
                if val is None:
                    return None
                if hasattr(val, 'item') and not isinstance(val, (str, bytes)):
                    try:
                        val = val.item()
                    except Exception:
                        pass
                try:
                    if pd.isna(val):
                        return None
                except Exception:
                    pass
                if key in ('symbol', 'date'):
                    return str(val)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            cols = ', '.join(values.keys())
            placeholders = ', '.join(['?' for _ in values])
            query = f"INSERT INTO features_finance ({cols}) VALUES ({placeholders})"

            try:
                final_values = tuple(_to_sql_param(k, values[k]) for k in values.keys())
                conn.execute(query, final_values)
            except Exception as e:
                logger.warning(f"Insert row failed: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} rows → features_finance")

    except Exception as e:
        logger.error(f"Lỗi khi lưu features_finance: {e}")
        import traceback
        traceback.print_exc()


def process_and_engineer_finance():
    """Process financial data: Clean + Feature Engineering (không normalize)"""
    logger.info("=" * 70)
    logger.info("[STAGE 2+3] CLEAN & ENGINEER FINANCE FEATURES")
    logger.info("=" * 70)

    # ============ STAGE 2: CLEAN ============
    logger.info("[STAGE 2] Làm sạch dữ liệu tài chính...")

    df = load_raw_from_database()
    if df is None:
        return None

    logger.info(f"Input: {len(df)} rows × {len(df.columns)} columns")
    logger.info(f"Sẽ giữ lại tối đa {TARGET_QUARTERS} quý cuối sau khi tính YoY")

    feature_cols = [
        'roe',
        'roa',
        'debt_to_equity',
        'net_profit_margin',
        'financial_leverage',
    ]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Thiếu cột: {missing_cols}")
        return None

    keep_cols = ['symbol', 'date'] + feature_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    if 'symbol' not in df.columns:
        df.insert(0, 'symbol', SYMBOL)

    logger.info(f"Selected columns: {df.columns.tolist()}")

    rows_before = len(df)
    df = df.dropna(subset=feature_cols, how='all')
    logger.info(f"Dropped null rows: {rows_before} → {len(df)} rows")

    # ============ STAGE 3: FEATURE ENGINEERING ============
    logger.info("[STAGE 3] Feature engineering...")

    # 1. Xử lý giá trị 0 (data quality) — thay bằng NaN trước khi fill
    for col in feature_cols:
        df.loc[df[col] == 0, col] = np.nan
    logger.info("Replaced 0 values with NaN")

    # 2. Forward fill rồi backward fill (phù hợp time series)
    for col in feature_cols:
        df[col] = df[col].ffill()
        df[col] = df[col].bfill()
    logger.info("Forward/backward filled NaN")

    # 3. Loại outliers (Z-score > 3)
    rows_before = len(df)
    for col in feature_cols:
        mask = df[col].notna()
        if mask.sum() > 2:
            z_scores = np.abs(stats.zscore(df.loc[mask, col]))
            outlier_mask = z_scores > 3
            if outlier_mask.any():
                df = df[~df.index.isin(df.loc[mask][outlier_mask].index)]
    logger.info(f"Outlier removal: {rows_before} → {len(df)} rows")

    # 4. Tính YoY (4-quarter lag = 1 năm)
    logger.info("Tính YoY changes (4Q lag)...")
    for col in ['roe', 'roa']:
        df[f'{col}_yoy'] = df[col].pct_change(periods=4) * 100  # phần trăm

    # Thay YoY = 0 bằng NaN rồi fill
    yoy_cols = ['roe_yoy', 'roa_yoy']
    for col in yoy_cols:
        df.loc[df[col] == 0, col] = np.nan
        df[col] = df[col].ffill()
        df[col] = df[col].bfill()

    # 5. Tính lag values (4 quý trước)
    for col in ['roe', 'roa']:
        df[f'{col}_lag4'] = df[col].shift(4)

    logger.info(f"Raw features: {feature_cols}")
    logger.info(f"YoY features: {[f'{c}_yoy' for c in ['roe', 'roa']]}")
    logger.info(f"Lag features: {[f'{c}_lag4' for c in ['roe', 'roa']]}")

    # 6. Giữ lại cửa sổ TARGET_QUARTERS gần nhất
    if len(df) > TARGET_QUARTERS:
        rows_before = len(df)
        df = df.tail(TARGET_QUARTERS).reset_index(drop=True)
        logger.info(f"Trimmed to target window: {rows_before} → {len(df)} rows")

    # NOTE: KHÔNG normalize ở đây.
    # Việc scale (MinMaxScaler) được thực hiện tập trung trong
    # BasePredictor.fit() — chỉ fit trên train set để tránh data leakage.
    logger.info("Skip normalization (sẽ scale tập trung trong BasePredictor.fit)")

    # 7. Lưu vào database
    logger.info("[OUTPUT] Lưu features đã xử lý → features_finance...")
    save_features_to_database(df)

    logger.info(f"Shape output: {df.shape}")
    logger.info(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    logger.info("=" * 70)
    logger.info("HOÀN THÀNH: Features đã được lưu vào features_finance")
    logger.info("=" * 70)

    return df


def process_finance():
    """Entry point dùng bởi `python -m preprocessing.process_finance` và pipeline."""
    return process_and_engineer_finance()


if __name__ == "__main__":
    process_finance()