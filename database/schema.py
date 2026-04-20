"""
Schema cho pipeline dữ liệu: Crawl → Clean → Engineer → Split
═══════════════════════════════════════════════════════════════
Chạy: python -m database.schema

Cấu trúc:
  STAGE 1: raw_finance (24 rows × 18+ columns từ vnstock Ratio API)
  STAGE 2+3: features_finance (24 rows × 8 columns sau clean & engineer)
  STAGE 4: train_features, val_features, test_features (split theo thời gian)
           train_weights (trọng số cho huấn luyện)
"""
from database.connection import get_connection
from loguru import logger


RAW_FINANCE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS raw_finance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        meta_ticker TEXT,
        meta_yearreport INTEGER,
        meta_lengthreport INTEGER,
        roe REAL,
        roa REAL,
        debt_to_equity REAL,
        fixed_asset_to_equity REAL,
        owners_equity_to_charter_capital REAL,
        net_profit_margin REAL,
        financial_leverage REAL,
        market_cap_bn_vnd REAL,
        outstanding_share_mil REAL,
        pe_ratio REAL,
        pb_ratio REAL,
        ps_ratio REAL,
        pcf_ratio REAL,
        eps_vnd REAL,
        bvps_vnd REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, date)
    )
"""

FEATURES_FINANCE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS features_finance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        roe REAL,
        roa REAL,
        debt_to_equity REAL,
        net_profit_margin REAL,
        financial_leverage REAL,
        roe_yoy REAL,
        roa_yoy REAL,
        roe_lag4 REAL,
        roa_lag4 REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, date)
    )
"""


def create_all_tables():
    """Tạo toàn bộ tables cho pipeline."""
    conn = get_connection()

    # =============================================
    # STAGE 0: RAW DATA GIÁ CỔ PHIếu
    # =============================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume REAL,
            symbol TEXT DEFAULT 'TCB',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS clean_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            sma_10 REAL, sma_20 REAL, sma_50 REAL,
            ema_12 REAL, ema_26 REAL,
            rsi_14 REAL,
            macd REAL, macd_signal REAL, macd_hist REAL,
            bb_upper REAL, bb_middle REAL, bb_lower REAL,
            atr_14 REAL, obv REAL,
            price_change REAL, price_change_5d REAL,
            volatility_10d REAL, volume_sma_10 REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date)
        )
    """)

    # =============================================
    # STAGE 0: RAW NEWS
    # =============================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            title TEXT,
            content TEXT,
            url TEXT,
            source TEXT,
            symbol TEXT DEFAULT 'TCB',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS clean_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            title TEXT,
            content TEXT,
            url TEXT,
            source TEXT,
            sentiment_neg REAL,
            sentiment_pos REAL,
            sentiment_neu REAL,
            sentiment_score REAL,
            sentiment_label TEXT,
            embedding_score REAL,
            embedding_label TEXT,
            daily_sentiment REAL,
            news_count INTEGER,
            embedding TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =============================================
    # STAGE 1: RAW FINANCE (và features_finance) — Từ vnstock Ratio API
    # =============================================

    conn.execute(RAW_FINANCE_TABLE_SQL)
    conn.execute(FEATURES_FINANCE_TABLE_SQL)

    # =============================================
    # STAGE 2: MERGED FEATURES — Input cho model
    # =============================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merged_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            sma_10 REAL, sma_20 REAL, sma_50 REAL,
            ema_12 REAL, ema_26 REAL,
            rsi_14 REAL,
            macd REAL, macd_signal REAL, macd_hist REAL,
            bb_upper REAL, bb_middle REAL, bb_lower REAL,
            atr_14 REAL, obv REAL,
            price_change REAL, price_change_5d REAL,
            volatility_10d REAL, volume_sma_10 REAL,
            roe REAL, roa REAL, debt_to_equity REAL,
            net_profit_margin REAL, financial_leverage REAL,
            roe_yoy REAL, roa_yoy REAL, roe_lag4 REAL, roa_lag4 REAL,
            daily_sentiment REAL,
            news_count INTEGER,
            target REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date)
        )
    """)

    # =============================================
    # SPLIT DATA — Train/Val/Test splits (finance)
    # =============================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS train_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            roe REAL, roa REAL, debt_to_equity REAL,
            net_profit_margin REAL, financial_leverage REAL,
            roe_yoy REAL, roa_yoy REAL, roe_lag4 REAL, roa_lag4 REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS val_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            roe REAL, roa REAL, debt_to_equity REAL,
            net_profit_margin REAL, financial_leverage REAL,
            roe_yoy REAL, roa_yoy REAL, roe_lag4 REAL, roa_lag4 REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            roe REAL, roa REAL, debt_to_equity REAL,
            net_profit_margin REAL, financial_leverage REAL,
            roe_yoy REAL, roa_yoy REAL, roe_lag4 REAL, roa_lag4 REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS train_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =============================================
    # MODEL OUTPUTS — Dự đoán và metrics
    # =============================================

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            model_name TEXT,
            predicted_price REAL,
            actual_price REAL,
            error_pct REAL,
            predicted_at DATETIME,
            updated_at DATETIME,
            UNIQUE(date, model_name)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE NOT NULL,
            rmse REAL, mae REAL, mape REAL,
            directional_accuracy REAL,
            train_loss REAL, val_loss REAL, test_loss REAL,
            epochs_trained INTEGER,
            trained_at DATETIME,
            is_best INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    logger.info("✅ Tất cả tables đã được tạo.")


def recreate_raw_finance_table():
    """Drop và tạo lại raw_finance để đồng bộ schema mới nhất."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS raw_finance")
    conn.execute(RAW_FINANCE_TABLE_SQL)
    conn.commit()
    conn.close()
    logger.info("✅ raw_finance đã được drop & recreate theo schema mới.")


def recreate_features_finance_table():
    """Drop và tạo lại features_finance để đồng bộ schema mới nhất."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS features_finance")
    conn.execute(FEATURES_FINANCE_TABLE_SQL)
    conn.commit()
    conn.close()
    logger.info("✅ features_finance đã được drop & recreate theo schema mới.")


def show_tables():
    """Hiển thị danh sách tables và số dòng."""
    conn = get_connection()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    print(f"\n📊 Database: {len(tables)} tables")
    print("=" * 55)

    stages = {
        "Giá cổ phiếu (Raw)": ["raw_prices"],
        "Giá cổ phiếu (Clean)": ["clean_prices"],
        "Tin tức (Raw)": ["raw_news"],
        "Tin tức (Clean + Sentiment)": ["clean_news"],
        "Tài chính (Raw)": ["raw_finance"],
        "Tài chính (Features)": ["features_finance"],
        "Merged (Input Model)": ["merged_features"],
        "Split Data (Finance)": ["train_features", "val_features", "test_features", "train_weights"],
        "Model Outputs": ["predictions", "model_metrics"],
    }

    existing_tables = {name[0] for name in tables}

    for stage, table_list in stages.items():
        stage_tables = [t for t in table_list if t in existing_tables]
        if stage_tables:
            print(f"\n{stage}:")
            for name in stage_tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                status = "✅" if count > 0 else "⬜"
                print(f"  {status} {name:<30} {count:>6} rows")

    print("\n" + "=" * 55)
    conn.close()


if __name__ == "__main__":
    create_all_tables()
    show_tables()