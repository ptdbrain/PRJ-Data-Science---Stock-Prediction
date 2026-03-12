"""
Thu thập giá lịch sử cổ phiếu TCB (3 năm).
═══════════════════════════════════════════
Phụ trách: Thành viên A (demo bởi nhóm trưởng)
Branch: feature/collect-prices
Chạy: python -m data_collection.collect_prices

Output: raw_prices table trong SQLite
        Columns: date, open, high, low, close, volume
"""
import pandas as pd
from vnstock import Vnstock
from loguru import logger
from database.connection import get_connection, read_table, write_table
from config.settings import SYMBOL, DATA_SOURCE, DATA_START_DATE, DATA_END_DATE

logger.add("logs/collect_prices.log", rotation="1 week")


def collect_prices():
    """
    Lấy giá lịch sử TCB từ vnstock API và lưu vào raw_prices table.

    Quy trình:
        1. Kết nối vnstock API
        2. Lấy data OHLCV từ DATA_START_DATE đến DATA_END_DATE
        3. Chuẩn hoá tên cột cho khớp database schema
        4. Kiểm tra data quality
        5. Lưu vào SQLite
    """
    logger.info(f"{'='*50}")
    logger.info(f"Thu thập giá {SYMBOL} | {DATA_START_DATE} → {DATA_END_DATE}")
    logger.info(f"{'='*50}")

    # ---- 1. Kết nối vnstock ----
    try:
        stock = Vnstock().stock(symbol=SYMBOL, source=DATA_SOURCE)
        logger.info(f"✅ Kết nối vnstock thành công (source: {DATA_SOURCE})")
    except Exception as e:
        logger.error(f"❌ Không kết nối được vnstock: {e}")
        raise

    # ---- 2. Lấy data lịch sử ----
    try:
        df = stock.quote.history(
            start=DATA_START_DATE,
            end=DATA_END_DATE,
            interval='1D'
        )
        logger.info(f"✅ Lấy được {len(df)} phiên giao dịch")
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy data: {e}")
        raise

    if df is None or df.empty:
        logger.error("❌ Không có data nào được trả về!")
        return

    # ---- 3. Chuẩn hoá tên cột ----
    # vnstock có thể trả về tên cột khác nhau tuỳ version
    # In ra để debug nếu cần
    logger.info(f"Columns gốc từ vnstock: {list(df.columns)}")

    # Mapping tên cột (vnstock3 thường dùng tiếng Anh lowercase)
    column_mapping = {
        'time': 'date',
        'Time': 'date',
        'Date': 'date',
        'date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
    }

    df = df.rename(columns=column_mapping)

    # Đảm bảo có đủ cột cần thiết
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Thiếu cột: {missing}")
        logger.error(f"Các cột hiện có: {list(df.columns)}")
        raise ValueError(f"Thiếu cột: {missing}")

    # Chỉ giữ cột cần thiết
    df = df[required_cols].copy()

    # ---- 4. Xử lý data ----
    # Chuyển date thành string format chuẩn
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Xoá duplicate (nếu có)
    before = len(df)
    df = df.drop_duplicates(subset='date', keep='last')
    if len(df) < before:
        logger.warning(f"⚠️  Xoá {before - len(df)} dòng duplicate")

    # Sort theo ngày
    df = df.sort_values('date').reset_index(drop=True)

    # ---- 5. Kiểm tra data quality ----
    logger.info(f"\n📊 DATA QUALITY CHECK:")
    logger.info(f"  Số phiên:    {len(df)}")
    logger.info(f"  Từ ngày:     {df['date'].iloc[0]}")
    logger.info(f"  Đến ngày:    {df['date'].iloc[-1]}")
    logger.info(f"  Giá min:     {df['close'].min():,.0f} VND")
    logger.info(f"  Giá max:     {df['close'].max():,.0f} VND")
    logger.info(f"  Giá cuối:    {df['close'].iloc[-1]:,.0f} VND")
    logger.info(f"  Volume TB:   {df['volume'].mean():,.0f}")

    # Kiểm tra null
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"⚠️  Có null values:\n{null_counts[null_counts > 0]}")
        # Drop rows có null
        df = df.dropna()
        logger.info(f"  Sau khi drop null: {len(df)} rows")

    # Kiểm tra giá trị bất thường
    if (df['close'] <= 0).any():
        logger.warning("⚠️  Có giá đóng cửa <= 0!")
    if (df['volume'] < 0).any():
        logger.warning("⚠️  Có volume < 0!")

    # ---- 6. Lưu vào database ----
    write_table(df, "raw_prices")
    logger.info(f"\n✅ Đã lưu {len(df)} phiên vào raw_prices table")

    # Verify bằng cách đọc lại
    verify = read_table("raw_prices")
    logger.info(f"✅ Verify: raw_prices có {len(verify)} rows trong database")

    return df


def show_sample():
    """Hiển thị vài dòng data để kiểm tra."""
    try:
        df = read_table("raw_prices")
        print(f"\n📋 raw_prices — {len(df)} rows")
        print("=" * 70)
        print("5 dòng ĐẦU TIÊN:")
        print(df.head().to_string(index=False))
        print("\n5 dòng CUỐI CÙNG:")
        print(df.tail().to_string(index=False))
        print("=" * 70)
    except Exception as e:
        print(f"❌ Chưa có data: {e}")


if __name__ == "__main__":
    collect_prices()
    show_sample()