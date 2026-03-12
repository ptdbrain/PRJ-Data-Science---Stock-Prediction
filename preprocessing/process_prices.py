"""
Tiền xử lý giá cổ phiếu + tính Technical Indicators.
═════════════════════════════════════════════════════
Phụ trách: Thành viên A (demo bởi nhóm trưởng)
Branch: feature/process-prices
Chạy: python -m preprocessing.process_prices

Input:  raw_prices table
Output: clean_prices table (giá đã clean + 18 technical indicators)
"""
import pandas as pd
import numpy as np
import ta
from loguru import logger
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

    # Xử lý missing values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"  Có {null_count} giá trị null")
        # Forward fill trước (dùng giá ngày trước), rồi backward fill
        df = df.ffill().bfill()
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            logger.warning(f"  Còn {remaining_nulls} null sau fill → drop")
            df = df.dropna()

    df = df.reset_index(drop=True)
    logger.info(f"  ✅ Data sạch: {len(df)} phiên giao dịch")

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
    # Simple Moving Averages — xu hướng ngắn/trung/dài hạn
    df['sma_10'] = ta.trend.sma_indicator(close, window=10)
    df['sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['sma_50'] = ta.trend.sma_indicator(close, window=50)

    # Exponential Moving Averages — nhạy hơn SMA với giá gần đây
    df['ema_12'] = ta.trend.ema_indicator(close, window=12)
    df['ema_26'] = ta.trend.ema_indicator(close, window=26)

    logger.info("  ✅ Trend: SMA(10,20,50), EMA(12,26)")

    # =====================
    # MOMENTUM INDICATORS
    # =====================
    # RSI — đo tốc độ thay đổi giá (overbought > 70, oversold < 30)
    df['rsi_14'] = ta.momentum.rsi(close, window=14)

    # MACD — tín hiệu mua/bán dựa trên EMA crossover
    macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    logger.info("  ✅ Momentum: RSI(14), MACD(12,26,9)")

    # =====================
    # VOLATILITY INDICATORS
    # =====================
    # Bollinger Bands — biên độ dao động giá
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()

    # ATR — mức dao động trung bình (đo volatility)
    df['atr_14'] = ta.volatility.average_true_range(high, low, close, window=14)

    logger.info("  ✅ Volatility: Bollinger Bands(20,2), ATR(14)")

    # =====================
    # VOLUME INDICATORS
    # =====================
    # OBV — xu hướng dòng tiền (volume xác nhận giá)
    df['obv'] = ta.volume.on_balance_volume(close, volume)

    logger.info("  ✅ Volume: OBV")

    # =====================
    # DERIVED FEATURES
    # =====================
    # Price change — biến động giá
    df['price_change'] = close.pct_change()           # % thay đổi 1 ngày
    df['price_change_5d'] = close.pct_change(periods=5)  # % thay đổi 5 ngày

    # Rolling volatility — độ biến động 10 ngày
    df['volatility_10d'] = close.rolling(window=10).std()

    # Volume trend — so sánh volume hiện tại với trung bình
    df['volume_sma_10'] = volume.rolling(window=10).mean()

    logger.info("  ✅ Derived: price_change(1d,5d), volatility(10d), volume_sma(10)")

    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 3: Xoá các dòng NaN ở đầu (do rolling windows cần N ngày warm-up).

    SMA_50 cần 50 ngày → 49 dòng đầu sẽ là NaN.
    Xoá tất cả dòng có bất kỳ NaN nào.
    """
    logger.info("--- Bước 3: Xoá warm-up rows ---")

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)

    logger.info(f"  Xoá {dropped} dòng warm-up (cần cho rolling windows)")
    logger.info(f"  ✅ Còn lại: {len(df)} dòng sạch")

    return df


def validate_output(df: pd.DataFrame) -> bool:
    """
    Bước 4: Kiểm tra chất lượng output trước khi lưu.
    """
    logger.info("--- Bước 4: Validate output ---")

    expected_cols = [
        'date', 'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'obv',
        'price_change', 'price_change_5d', 'volatility_10d', 'volume_sma_10'
    ]

    # Kiểm tra có đủ cột
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"  ❌ Thiếu cột: {missing_cols}")
        return False

    # Kiểm tra không có NaN
    null_count = df[expected_cols].isnull().sum().sum()
    if null_count > 0:
        logger.error(f"  ❌ Còn {null_count} giá trị NaN")
        return False

    # Kiểm tra RSI trong khoảng hợp lệ
    if (df['rsi_14'] < 0).any() or (df['rsi_14'] > 100).any():
        logger.warning("  ⚠️ RSI ngoài khoảng [0, 100]")

    # Kiểm tra Bollinger Bands logic: upper > middle > lower
    bb_valid = (df['bb_upper'] >= df['bb_middle']).all() and \
               (df['bb_middle'] >= df['bb_lower']).all()
    if not bb_valid:
        logger.warning("  ⚠️ Bollinger Bands không hợp lệ (upper < middle hoặc middle < lower)")

    logger.info(f"  ✅ Validate passed: {len(df)} rows, {len(expected_cols)} columns, 0 NaN")
    return True


def process_prices():
    """
    Pipeline chính: raw_prices → clean_prices.
    Chạy 4 bước tuần tự.
    """
    logger.info(f"{'='*60}")
    logger.info("TIỀN XỬ LÝ GIÁ CỔ PHIẾU TCB")
    logger.info(f"{'='*60}")

    # Đọc raw data
    try:
        df = read_table("raw_prices")
    except Exception as e:
        logger.error(f"❌ Chưa có raw_prices! Chạy collect_prices trước: {e}")
        return None

    if df.empty:
        logger.error("❌ raw_prices trống! Chạy: python -m data_collection.collect_prices")
        return None

    logger.info(f"Input: {len(df)} rows từ raw_prices")

    # Pipeline
    df = clean_raw_prices(df)
    df = add_technical_indicators(df)
    df = drop_warmup_rows(df)

    # Chuyển date về string trước khi lưu
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Validate
    if not validate_output(df):
        logger.error("❌ Validation failed — không lưu")
        return None

    # Lưu
    write_table(df, "clean_prices")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ HOÀN THÀNH")
    logger.info(f"  Input:     raw_prices")
    logger.info(f"  Output:    clean_prices ({len(df)} rows, {df.shape[1]} columns)")
    logger.info(f"  Khoảng:    {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    logger.info(f"  Features:  {df.shape[1] - 6} technical indicators + 6 OHLCV")
    logger.info(f"{'='*60}")

    return df


def show_sample():
    """Hiển thị sample data để kiểm tra nhanh."""
    try:
        df = read_table("clean_prices")
        print(f"\n📋 clean_prices — {len(df)} rows × {df.shape[1]} columns")
        print("=" * 80)

        print("\n🔢 Columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:>2}. {col}")

        print(f"\n📊 5 dòng cuối:")
        print(df.tail().to_string(index=False))

        print(f"\n📈 Thống kê indicators (5 dòng cuối):")
        indicator_cols = ['rsi_14', 'macd', 'bb_upper', 'bb_middle', 'bb_lower', 'atr_14']
        available = [c for c in indicator_cols if c in df.columns]
        if available:
            print(df[['date'] + available].tail().to_string(index=False))

        print("=" * 80)
    except Exception as e:
        print(f"❌ Chưa có data: {e}")


if __name__ == "__main__":
    process_prices()
    show_sample()