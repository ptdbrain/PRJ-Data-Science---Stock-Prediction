"""
Process financial data: clean + feature engineering without look-ahead bias.
"""
import os
import sys

import numpy as np
import pandas as pd
from utils.logger import logger
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SYMBOL, TABLE_FEATURES_FINANCE
from database.connection import get_connection, write_table
from preprocessing.finance_utils import (
    normalize_quarter_code,
    quarter_effective_date,
    quarter_period_end_date,
)


logger.add("logs/process_finance.log", rotation="1 week")

TARGET_QUARTERS = 24


def load_raw_from_database():
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
    except Exception as exc:
        logger.error(f"Lỗi khi đọc raw_finance: {exc}")
        return None


def save_features_to_database(df):
    write_table(df, TABLE_FEATURES_FINANCE, if_exists="replace")
    logger.info(f"Saved {len(df)} rows → {TABLE_FEATURES_FINANCE}")


def _prepare_finance_dates(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["date"] = prepared["date"].apply(normalize_quarter_code)

    if "period_end_date" not in prepared.columns:
        prepared["period_end_date"] = prepared["date"].apply(
            lambda value: quarter_period_end_date(value).isoformat()
        )
    else:
        prepared["period_end_date"] = pd.to_datetime(prepared["period_end_date"]).dt.strftime("%Y-%m-%d")

    if "effective_date" not in prepared.columns:
        prepared["effective_date"] = prepared["date"].apply(
            lambda value: quarter_effective_date(value).isoformat()
        )
    else:
        prepared["effective_date"] = pd.to_datetime(prepared["effective_date"]).dt.strftime("%Y-%m-%d")

    prepared = prepared.sort_values(["period_end_date", "effective_date", "date"]).reset_index(drop=True)
    return prepared


def process_and_engineer_finance():
    logger.info("=" * 70)
    logger.info("[STAGE 2+3] CLEAN & ENGINEER FINANCE FEATURES")
    logger.info("=" * 70)

    df = load_raw_from_database()
    if df is None:
        return None

    feature_cols = [
        "roe",
        "roa",
        "debt_to_equity",
        "net_profit_margin",
        "financial_leverage",
        "eps",    # Earnings Per Share — cột mới
    ]
    # eps là cột tùy chọn (có thể thiếu trong dữ liệu cũ)
    base_required = ["date", "roe", "roa", "debt_to_equity", "net_profit_margin", "financial_leverage"]
    missing_cols = [column for column in base_required if column not in df.columns]
    if missing_cols:
        logger.error(f"Thiếu cột: {missing_cols}")
        return None

    if "eps" not in df.columns:
        if "eps_vnd" in df.columns:
            logger.info("Mapping 'eps_vnd' -> 'eps'")
            df["eps"] = df["eps_vnd"]
        else:
            logger.warning("Cột 'eps'/'eps_vnd' không có trong raw_finance — bỏ qua EPS features")
            feature_cols = [c for c in feature_cols if c != "eps"]

    df = _prepare_finance_dates(df)
    keep_cols = ["symbol", "date", "period_end_date", "effective_date"] + feature_cols
    if "symbol" not in df.columns:
        df.insert(0, "symbol", SYMBOL)
    df = df[keep_cols].copy()

    rows_before = len(df)
    df = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    logger.info(f"Dropped null rows: {rows_before} → {len(df)} rows")

    for column in feature_cols:
        df.loc[df[column] == 0, column] = np.nan
        df[column] = df[column].ffill()
    logger.info("Forward-filled missing finance values without backward fill")

    rows_before = len(df)
    for column in feature_cols:
        mask = df[column].notna()
        if mask.sum() > 2:
            z_scores = np.abs(stats.zscore(df.loc[mask, column]))
            outlier_mask = z_scores > 3
            if outlier_mask.any():
                df = df[~df.index.isin(df.loc[mask][outlier_mask].index)]
    df = df.reset_index(drop=True)
    logger.info(f"Outlier removal: {rows_before} → {len(df)} rows")

    for column in ["roe", "roa"]:
        df[f"{column}_yoy"] = df[column].pct_change(periods=4) * 100
        df[f"{column}_lag4"] = df[column].shift(4)

    # EPS YoY (tăng trưởng EPS so với cùng quý năm trước)
    if "eps" in df.columns:
        df["eps_yoy"] = df["eps"].pct_change(periods=4) * 100
        df["eps_yoy"] = df["eps_yoy"].replace([np.inf, -np.inf], np.nan)

    for column in ["roe_yoy", "roa_yoy"]:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)

    if len(df) > TARGET_QUARTERS:
        rows_before = len(df)
        df = df.tail(TARGET_QUARTERS).reset_index(drop=True)
        logger.info(f"Trimmed to target window: {rows_before} → {len(df)} rows")

    logger.info("[OUTPUT] Lưu features đã xử lý → features_finance...")
    save_features_to_database(df)

    logger.info(f"Shape output: {df.shape}")
    logger.info(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    logger.info("=" * 70)
    logger.info("HOÀN THÀNH: Features đã được lưu vào features_finance")
    logger.info("=" * 70)

    return df


def process_finance():
    return process_and_engineer_finance()


if __name__ == "__main__":
    process_finance()
