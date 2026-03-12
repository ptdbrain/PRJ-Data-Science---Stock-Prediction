"""
Tiền xử lý giá cổ phiếu + tính Technical Indicators.
═════════════════════════════════════════════════════
Phụ trách: Thành viên A
Branch: feature/process-prices
Chạy: python -m preprocessing.process_prices

Input:  raw_prices table
Output: clean_prices table (có thêm technical indicators)
"""
import pandas as pd
import ta
from loguru import logger
from database.connection import read_table, write_table

logger.add("logs/process_prices.log", rotation="1 week")


def process_prices():
    """
    Đọc raw_prices → xử lý missing values → tính technical indicators → lưu clean_prices.

    Technical indicators cần tính:
    - SMA (10, 20, 50)
    - EMA (12, 26)
    - RSI (14)
    - MACD (macd, signal, histogram)
    - Bollinger Bands (upper, middle, lower)
    - ATR (14)
    - OBV
    - Price change (1d, 5d)
    - Volatility (10d rolling std)
    - Volume SMA (10)
    """
    logger.info("Tiền xử lý giá cổ phiếu...")

    # Đọc raw data
    df = read_table("raw_prices")
    logger.info(f"Loaded {len(df)} rows từ raw_prices")

    # TODO: Thành viên A implement
    # Gợi ý:
    # 1. Sort theo date
    # 2. Xử lý missing values (dropna hoặc fillna)
    # 3. Tính technical indicators bằng thư viện `ta`:
    #    - df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    #    - df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    #    - macd = ta.trend.MACD(df['close'])
    #    - bb = ta.volatility.BollingerBands(df['close'])
    #    - ... (xem config/settings.py → TECHNICAL_FEATURES)
    # 4. Drop rows NaN ở đầu (do rolling windows)
    # 5. Lưu: write_table(df, "clean_prices")

    raise NotImplementedError("Thành viên A cần implement hàm này")


if __name__ == "__main__":
    process_prices()
