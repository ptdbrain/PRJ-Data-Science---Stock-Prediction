"""
Tạo tất cả tables trong database.
═══════════════════════════════════
Chạy: python -m database.schema

Mỗi thành viên chạy 1 lần sau khi clone repo.
Chạy lại nếu cần thêm table mới (không mất data cũ nhờ IF NOT EXISTS).
"""
from database.connection import get_connection
from loguru import logger


def create_all_tables():
    """Tạo toàn bộ tables cho project."""
    conn = get_connection()

    # =========================================
    # RAW TABLES — Data gốc, không chỉnh sửa
    # =========================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_finance (
            quarter TEXT,
            year INTEGER,
            quarter_num INTEGER,
            report_type TEXT,
            metric_name TEXT,
            value REAL,
            PRIMARY KEY (quarter, report_type, metric_name)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            title TEXT,
            content TEXT,
            url TEXT UNIQUE,
            source TEXT
        )
    """)

    # =========================================
    # CLEAN TABLES — Đã tiền xử lý
    # =========================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS clean_prices (
            date TEXT PRIMARY KEY,
            open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            sma_10 REAL, sma_20 REAL, sma_50 REAL,
            ema_12 REAL, ema_26 REAL,
            rsi_14 REAL,
            macd REAL, macd_signal REAL, macd_hist REAL,
            bb_upper REAL, bb_middle REAL, bb_lower REAL,
            atr_14 REAL, obv REAL,
            price_change REAL, price_change_5d REAL,
            volatility_10d REAL, volume_sma_10 REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS clean_finance (
            quarter TEXT PRIMARY KEY,
            revenue REAL,
            net_income REAL,
            total_assets REAL,
            total_equity REAL,
            roe REAL,
            roa REAL,
            nim REAL,
            pe_ratio REAL,
            pb_ratio REAL,
            debt_to_equity REAL,
            npl_ratio REAL,
            cost_to_income REAL,
            revenue_growth REAL,
            profit_growth REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS clean_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            title TEXT,
            sentiment_score REAL,
            sentiment_pos REAL,
            sentiment_neg REAL,
            sentiment_neu REAL
        )
    """)

    # =========================================
    # MERGED FEATURES — Input cho model
    # =========================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merged_features (
            date TEXT PRIMARY KEY,
            open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            sma_10 REAL, sma_20 REAL, sma_50 REAL,
            ema_12 REAL, ema_26 REAL,
            rsi_14 REAL,
            macd REAL, macd_signal REAL, macd_hist REAL,
            bb_upper REAL, bb_middle REAL, bb_lower REAL,
            atr_14 REAL, obv REAL,
            price_change REAL, price_change_5d REAL,
            volatility_10d REAL, volume_sma_10 REAL,
            roe REAL, roa REAL, nim REAL,
            pe_ratio REAL, pb_ratio REAL,
            debt_to_equity REAL, revenue_growth REAL, profit_growth REAL,
            daily_sentiment REAL,
            news_count INTEGER,
            target REAL
        )
    """)

    # =========================================
    # OUTPUT TABLES
    # =========================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            date TEXT,
            model_name TEXT,
            predicted_price REAL,
            actual_price REAL,
            predicted_at TEXT,
            PRIMARY KEY (date, model_name)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            model_name TEXT PRIMARY KEY,
            rmse REAL,
            mae REAL,
            mape REAL,
            directional_accuracy REAL,
            train_loss REAL,
            val_loss REAL,
            test_loss REAL,
            epochs_trained INTEGER,
            trained_at TEXT,
            is_best INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    logger.info("✅ Tất cả tables đã được tạo.")


def show_tables():
    """Hiển thị danh sách tables và số dòng."""
    conn = get_connection()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    print(f"\n📊 Database: {len(tables)} tables")
    print("=" * 40)
    for (name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        status = "✅" if count > 0 else "⬜"
        print(f"  {status} {name:<25} {count:>6} rows")
    print("=" * 40)
    conn.close()


if __name__ == "__main__":
    create_all_tables()
    show_tables()
