"""
Thu thập giá lịch sử cổ phiếu TCB (3 năm).
═══════════════════════════════════════════
Phụ trách: Thành viên A
Branch: feature/collect-prices
Chạy: python -m data_collection.collect_prices

Output: raw_prices table trong SQLite
"""
from vnstock import Vnstock
from loguru import logger
from database.connection import get_connection, write_table
from config.settings import SYMBOL, DATA_SOURCE, DATA_START_DATE, DATA_END_DATE

logger.add("logs/collect_prices.log", rotation="1 week")


def collect_prices():
    """
    Lấy giá lịch sử TCB từ vnstock và lưu vào raw_prices.

    Columns cần có: date, open, high, low, close, volume
    """
    logger.info(f"Thu thập giá {SYMBOL} từ {DATA_START_DATE} đến {DATA_END_DATE}...")

    # TODO: Thành viên A implement
    # Gợi ý:
    # 1. Dùng Vnstock().stock(symbol=SYMBOL, source=DATA_SOURCE)
    # 2. Gọi stock.quote.history(start=..., end=..., interval='1D')
    # 3. Rename columns cho khớp schema: date, open, high, low, close, volume
    # 4. Lưu bằng write_table(df, "raw_prices")

    raise NotImplementedError("Thành viên A cần implement hàm này")


if __name__ == "__main__":
    collect_prices()
