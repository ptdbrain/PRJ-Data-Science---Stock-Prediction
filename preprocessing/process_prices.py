"""
Tiền xử lý giá cổ phiếu + tính Technical Indicators.
═════════════════════════════════════════════════════
Phụ trách: Thành viên A (demo bởi nhóm trưởng)
Branch: feature/process-prices
Chạy: python -m preprocessing.process_prices

Input:  raw_prices table
Output: clean_prices table (giá đã clean + technical indicators + stationary features)
"""
import pandas as pd
import numpy as np
import ta
from utils.logger import logger
from database.connection import read_table, write_table

logger.add("logs/process_prices.log", rotation="1 week")


def clean_raw_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 1: Làm sạch data giá thô.

    Xử lý:
    - Sort theo ngày
    - Xoá duplicate
    - Xử lý missing values
    - Loại bỏ giá trị bất thường (giá <= 0, volume < 0)
    """
    logger.info("--- Bước 1: Làm sạch data thô ---")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Xoá duplicate
    before = len(df)
    df = df.drop_duplicates(subset='date', keep='last')
    if len(df) < before:
        logger.warning(f"  Xoá {before - len(df)} dòng duplicate")

    # Loại bỏ giá trị bất thường
    invalid_mask = (
        (df['close'] <= 0) | (df['open'] <= 0) |
        (df['high'] <= 0) | (df['low'] <= 0) |
        (df['volume'] < 0)
    )
    if invalid_mask.any():
        logger.warning(f"  Loại bỏ {invalid_mask.sum()} dòng có giá trị bất thường")
        df = df[~invalid_mask]

    required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_ohlcv_mask = df[required_ohlcv_cols].isnull().any(axis=1)
    if missing_ohlcv_mask.any():
        logger.warning(
            f"  Loại bỏ {missing_ohlcv_mask.sum()} dòng thiếu OHLCV trọng yếu "
            "(không forward/backward fill dữ liệu thị trường)"
        )
        df = df[~missing_ohlcv_mask]

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"  Còn {remaining_nulls} null ngoài OHLCV -> drop")
        df = df.dropna()

    df = df.reset_index(drop=True)
    logger.info(f"  Data sach: {len(df)} phien giao dich")

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 2: Tính tất cả technical indicators.

    Nhóm indicators:
    - Trend: SMA, EMA
    - Momentum: RSI, MACD
    - Volatility: Bollinger Bands, ATR
    - Volume: OBV
    - Derived: Price change, Volatility, Volume trend
    """
    logger.info("--- Bước 2: Tính Technical Indicators ---")

    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume'].astype(float)

    # =====================
    # TREND INDICATORS
    # =====================
    # Simple Moving Averages
    df['sma_10'] = ta.trend.sma_indicator(close, window=10)
    df['sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['sma_50'] = ta.trend.sma_indicator(close, window=50)

    # Exponential Moving Averages
    df['ema_12'] = ta.trend.ema_indicator(close, window=12)
    df['ema_26'] = ta.trend.ema_indicator(close, window=26)

    logger.info("  Trend: SMA(10,20,50), EMA(12,26)")

    # =====================
    # MOMENTUM INDICATORS
    # =====================
    # RSI
    df['rsi_14'] = ta.momentum.rsi(close, window=14)

    # MACD
    macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    logger.info("  Momentum: RSI(14), MACD(12,26,9)")

    # =====================
    # VOLATILITY INDICATORS
    # =====================
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()

    # ATR
    df['atr_14'] = ta.volatility.average_true_range(high, low, close, window=14)

    logger.info("  Volatility: Bollinger Bands(20,2), ATR(14)")

    # =====================
    # VOLUME INDICATORS
    # =====================
    # OBV
    df['obv'] = ta.volume.on_balance_volume(close, volume)

    logger.info("  Volume: OBV")

    # =====================
    # DERIVED FEATURES
    # =====================
    # Price change
    df['price_change'] = close.pct_change()
    df['price_change_5d'] = close.pct_change(periods=5)

    # Rolling volatility
    df['volatility_10d'] = close.rolling(window=10).std()

    # Volume trend
    df['volume_sma_10'] = volume.rolling(window=10).mean()

    logger.info("  Derived: price_change(1d,5d), volatility(10d), volume_sma(10)")

    return df


def make_features_stationary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 2b: Chuyển đổi features phi dừng (non-stationary) sang dạng tỷ lệ.

    Vấn đề: Giá cổ phiếu năm 2021 (~20k) khác hoàn toàn năm 2024 (~40k).
    Scaler fit trên train sẽ bị lệch hoàn toàn khi transform test.

    Giải pháp: Chuyển tất cả giá trị tuyệt đối sang phần trăm / tỷ lệ:
    - OHLC prices -> Log Returns
    - SMA/EMA -> khoảng cách % so với close hiện tại
    - Bollinger Bands -> vị trí trong band (0-1) + bandwidth
    - ATR -> tỷ lệ % so với giá
    - OBV -> thay đổi %
    - Volume -> tỷ lệ so với trung bình
    """
    logger.info("--- Buoc 2b: Chuyen features sang dang stationary ---")

    df = df.copy()
    close = df['close']

    # 1. Giá OHLC -> Log Returns
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_ret'] = np.log(df[col] / df[col].shift(1))
    logger.info("  Log Returns: open_ret, high_ret, low_ret, close_ret")

    # 2. Volume -> tỷ lệ so với trung bình 10 ngày
    vol_sma = df['volume_sma_10'].replace(0, np.nan)
    df['volume_ratio'] = df['volume'] / vol_sma
    logger.info("  Volume ratio vs SMA10")

    # 3. SMA/EMA -> khoảng cách % so với giá hiện tại
    for ma_col in ['sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26']:
        ma_val = df[ma_col].replace(0, np.nan)
        df[f'{ma_col}_dist'] = (close - df[ma_col]) / ma_val
    logger.info("  SMA/EMA distance: sma_10_dist ... ema_26_dist")

    # 4. Bollinger Bands -> vị trí trong band (0-1) + bandwidth
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (close - df['bb_lower']) / bb_range.replace(0, np.nan)
    bb_mid = df['bb_middle'].replace(0, np.nan)
    df['bb_width'] = bb_range / bb_mid
    logger.info("  Bollinger: bb_position, bb_width")

    # 5. ATR -> tỷ lệ % so với giá
    df['atr_pct'] = df['atr_14'] / close.replace(0, np.nan)
    logger.info("  ATR pct of close")

    # 6. OBV -> thay đổi %
    df['obv_change'] = df['obv'].pct_change()
    df['obv_change'] = df['obv_change'].replace([np.inf, -np.inf], np.nan)
    logger.info("  OBV change pct")

    # 7. Volatility -> tỷ lệ % so với giá
    df['volatility_pct'] = df['volatility_10d'] / close.replace(0, np.nan)
    logger.info("  Volatility pct of close")

    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 3: Xoá các dòng NaN ở đầu (do rolling windows cần N ngày warm-up).

    SMA_50 cần 50 ngày -> 49 dòng đầu sẽ là NaN.
    Xoá tất cả dòng có bất kỳ NaN nào.
    """
    logger.info("--- Bước 3: Xoá warm-up rows ---")

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)

    logger.info(f"  Xoa {dropped} dong warm-up (can cho rolling windows)")
    logger.info(f"  Con lai: {len(df)} dong sach")

    return df


def validate_output(df: pd.DataFrame) -> bool:
    """
    Bước 4: Kiểm tra chất lượng output trước khi lưu.
    """
    logger.info("--- Bước 4: Validate output ---")

    # Cột gốc + cột stationary mới
    expected_cols = [
        'date', 'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'obv',
        'price_change', 'price_change_5d', 'volatility_10d', 'volume_sma_10',
        # Stationary features
        'close_ret', 'open_ret', 'high_ret', 'low_ret',
        'volume_ratio',
        'sma_10_dist', 'sma_20_dist', 'sma_50_dist', 'ema_12_dist', 'ema_26_dist',
        'bb_position', 'bb_width',
        'atr_pct', 'obv_change', 'volatility_pct',
    ]

    # Kiểm tra có đủ cột
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"  Thieu cot: {missing_cols}")
        return False

    # Kiểm tra không có NaN
    null_count = df[expected_cols].isnull().sum().sum()
    if null_count > 0:
        logger.error(f"  Con {null_count} gia tri NaN")
        return False

    # Kiểm tra RSI trong khoảng hợp lệ
    if (df['rsi_14'] < 0).any() or (df['rsi_14'] > 100).any():
        logger.warning("  RSI ngoai khoang [0, 100]")

    # Kiểm tra Bollinger Bands logic: upper > middle > lower
    bb_valid = (df['bb_upper'] >= df['bb_middle']).all() and \
               (df['bb_middle'] >= df['bb_lower']).all()
    if not bb_valid:
        logger.warning("  Bollinger Bands khong hop le (upper < middle hoac middle < lower)")

    logger.info(f"  Validate passed: {len(df)} rows, {len(expected_cols)} columns, 0 NaN")
    return True


def process_prices():
    """
    Pipeline chính: raw_prices -> clean_prices.
    Chạy 4 bước tuần tự.
    """
    logger.info(f"{'='*60}")
    logger.info("TIEN XU LY GIA CO PHIEU TCB")
    logger.info(f"{'='*60}")

    # Đọc raw data
    try:
        df = read_table("raw_prices")
    except Exception as e:
        logger.error(f"Chua co raw_prices! Chay collect_prices truoc: {e}")
        return None

    if df.empty:
        logger.error("raw_prices trong! Chay: python -m data_collection.collect_prices")
        return None

    logger.info(f"Input: {len(df)} rows tu raw_prices")

    # Pipeline
    df = clean_raw_prices(df)
    df = add_technical_indicators(df)
    df = make_features_stationary(df)    # <-- NEW: stationary features
    df = drop_warmup_rows(df)

    # Chuyển date về string trước khi lưu
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Validate
    if not validate_output(df):
        logger.error("Validation failed -- khong luu")
        return None

    # Lưu
    write_table(df, "clean_prices")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"HOAN THANH")
    logger.info(f"  Input:     raw_prices")
    logger.info(f"  Output:    clean_prices ({len(df)} rows, {df.shape[1]} columns)")
    logger.info(f"  Khoang:    {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    logger.info(f"  Features:  {df.shape[1] - 6} technical + stationary indicators + 6 OHLCV")
    logger.info(f"{'='*60}")

    return df


def show_sample():
    """Hiển thị sample data để kiểm tra nhanh."""
    try:
        df = read_table("clean_prices")
        print(f"\nclean_prices -- {len(df)} rows x {df.shape[1]} columns")
        print("=" * 80)

        print("\nColumns:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:>2}. {col}")

        print(f"\n5 dong cuoi:")
        print(df.tail().to_string(index=False))

        print(f"\nThong ke indicators (5 dong cuoi):")
        indicator_cols = ['rsi_14', 'macd', 'bb_position', 'bb_width', 'atr_pct', 'close_ret']
        available = [c for c in indicator_cols if c in df.columns]
        if available:
            print(df[['date'] + available].tail().to_string(index=False))

        print("=" * 80)
    except Exception as e:
        print(f"Chua co data: {e}")


if __name__ == "__main__":
    process_prices()
    show_sample()
