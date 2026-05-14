"""
Schema cho pipeline dữ liệu: Crawl → Clean → Engineer → Split
"""
from datetime import datetime

from utils.logger import logger

from database.connection import get_connection


RAW_PRICES_TABLE_SQL = """
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
"""

CLEAN_PRICES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS clean_prices (
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
        -- Stationary features (relative / percentage)
        close_ret REAL, open_ret REAL, high_ret REAL, low_ret REAL,
        volume_ratio REAL,
        sma_10_dist REAL, sma_20_dist REAL, sma_50_dist REAL,
        ema_12_dist REAL, ema_26_dist REAL,
        bb_position REAL, bb_width REAL,
        atr_pct REAL, obv_change REAL, volatility_pct REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(date)
    )
"""

RAW_NEWS_TABLE_SQL = """
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
"""

CLEAN_NEWS_TABLE_SQL = """
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
        tfidf_sentiment REAL,          -- Classical NLP score
        embedding_score REAL,
        embedding_label TEXT,
        daily_sentiment REAL,
        daily_tfidf_sentiment REAL,    -- Daily avg classical NLP
        news_count INTEGER,
        embedding TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

DAILY_NEWS_EMBEDDINGS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS daily_news_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        embedding_mean TEXT,
        embedding_std TEXT,
        embedding_score_mean REAL,
        embedding_score_std REAL,
        news_count INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(date)
    )
"""

RAW_FINANCE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS raw_finance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        period_end_date TEXT NOT NULL,
        effective_date TEXT NOT NULL,
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
        period_end_date TEXT NOT NULL,
        effective_date TEXT NOT NULL,
        roe REAL,
        roa REAL,
        debt_to_equity REAL,
        net_profit_margin REAL,
        financial_leverage REAL,
        eps REAL,                  -- Earnings Per Share
        roe_yoy REAL,
        roa_yoy REAL,
        roe_lag4 REAL,
        roa_lag4 REAL,
        eps_yoy REAL,              -- EPS YoY growth
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, date)
    )
"""

MERGED_FEATURES_TABLE_SQL = """
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
        -- Stationary features (relative / percentage)
        close_ret REAL, open_ret REAL, high_ret REAL, low_ret REAL,
        volume_ratio REAL,
        sma_10_dist REAL, sma_20_dist REAL, sma_50_dist REAL,
        ema_12_dist REAL, ema_26_dist REAL,
        bb_position REAL, bb_width REAL,
        atr_pct REAL, obv_change REAL, volatility_pct REAL,
        -- Finance features
        roe REAL, roa REAL, debt_to_equity REAL,
        net_profit_margin REAL, financial_leverage REAL,
        eps REAL, eps_yoy REAL,
        roe_yoy REAL, roa_yoy REAL, roe_lag4 REAL, roa_lag4 REAL,
        -- Sentiment features
        daily_sentiment REAL,
        news_count INTEGER,
        embedding_score_mean REAL,
        embedding_score_std REAL,
        tfidf_sentiment REAL,
        -- Target & display
        target INTEGER,
        next_close REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(date)
    )
"""

TRAIN_FEATURES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS train_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        roe REAL, roa REAL, debt_to_equity REAL,
        net_profit_margin REAL, financial_leverage REAL,
        roe_yoy REAL, roa_yoy REAL, roe_lag4 REAL, roa_lag4 REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

VAL_FEATURES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS val_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        roe REAL, roa REAL, debt_to_equity REAL,
        net_profit_margin REAL, financial_leverage REAL,
        roe_yoy REAL, roa_yoy REAL, roe_lag4 REAL, roa_lag4 REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

TEST_FEATURES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS test_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        roe REAL, roa REAL, debt_to_equity REAL,
        net_profit_margin REAL, financial_leverage REAL,
        roe_yoy REAL, roa_yoy REAL, roe_lag4 REAL, roa_lag4 REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

TRAIN_WEIGHTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS train_weights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        row_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        weight REAL NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

PREDICTIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        model_name TEXT,
        predicted_proba REAL,          -- P(TĂNG) ∈ [0, 1]
        predicted_trend INTEGER,       -- 1=TĂNG, 0=GIẢM
        actual_trend INTEGER,          -- Ground truth (nếu đã biết)
        predicted_at DATETIME,
        updated_at DATETIME,
        UNIQUE(date, model_name)
    )
"""

MODEL_METRICS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT UNIQUE NOT NULL,
        -- Classification metrics
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL,
        roc_auc REAL,
        -- Training info
        val_loss REAL,
        test_loss REAL,
        epochs_trained INTEGER,
        trained_at DATETIME,
        train_end_date TEXT,
        val_end_date TEXT,
        lookback_days INTEGER,
        is_best INTEGER DEFAULT 0
    )
"""

MANAGED_TABLE_SQL = {
    "raw_prices": RAW_PRICES_TABLE_SQL,
    "clean_prices": CLEAN_PRICES_TABLE_SQL,
    "raw_news": RAW_NEWS_TABLE_SQL,
    "clean_news": CLEAN_NEWS_TABLE_SQL,
    "daily_news_embeddings": DAILY_NEWS_EMBEDDINGS_TABLE_SQL,
    "raw_finance": RAW_FINANCE_TABLE_SQL,
    "features_finance": FEATURES_FINANCE_TABLE_SQL,
    "merged_features": MERGED_FEATURES_TABLE_SQL,
    "train_features": TRAIN_FEATURES_TABLE_SQL,
    "val_features": VAL_FEATURES_TABLE_SQL,
    "test_features": TEST_FEATURES_TABLE_SQL,
    "train_weights": TRAIN_WEIGHTS_TABLE_SQL,
    "predictions": PREDICTIONS_TABLE_SQL,
    "model_metrics": MODEL_METRICS_TABLE_SQL,
}


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _get_table_columns(conn, table_name: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]


def _get_expected_columns(table_name: str) -> list[str]:
    create_sql = MANAGED_TABLE_SQL[table_name]
    temp_conn = get_connection(":memory:")
    try:
        temp_conn.execute(create_sql)
        return _get_table_columns(temp_conn, table_name)
    finally:
        temp_conn.close()


def ensure_managed_table_schema(conn, table_name: str) -> None:
    if table_name not in MANAGED_TABLE_SQL:
        return

    create_sql = MANAGED_TABLE_SQL[table_name]
    if not _table_exists(conn, table_name):
        conn.execute(create_sql)
        return

    actual_columns = _get_table_columns(conn, table_name)
    expected_columns = _get_expected_columns(table_name)
    if actual_columns == expected_columns:
        return

    backup_name = f"{table_name}__backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.warning(
        f"Schema mismatch detected for '{table_name}'. "
        f"Backing up to '{backup_name}' and recreating canonical schema."
    )

    conn.execute(f"ALTER TABLE {table_name} RENAME TO {backup_name}")
    conn.execute(create_sql)

    common_columns = [column for column in actual_columns if column in expected_columns]
    if common_columns:
        quoted_columns = ", ".join(common_columns)
        conn.execute(
            f"INSERT INTO {table_name} ({quoted_columns}) "
            f"SELECT {quoted_columns} FROM {backup_name}"
        )


def create_all_tables():
    """Tạo toàn bộ bảng cho pipeline và reconcile schema managed tables."""
    conn = get_connection()
    try:
        for create_sql in MANAGED_TABLE_SQL.values():
            conn.execute(create_sql)

        for table_name in MANAGED_TABLE_SQL:
            ensure_managed_table_schema(conn, table_name)

        conn.commit()
    finally:
        conn.close()

    logger.info("✅ Tất cả tables đã được tạo và reconcile schema.")


def recreate_raw_finance_table():
    conn = get_connection()
    try:
        conn.execute("DROP TABLE IF EXISTS raw_finance")
        conn.execute(RAW_FINANCE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()
    logger.info("✅ raw_finance đã được drop & recreate theo schema mới.")


def recreate_features_finance_table():
    conn = get_connection()
    try:
        conn.execute("DROP TABLE IF EXISTS features_finance")
        conn.execute(FEATURES_FINANCE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()
    logger.info("✅ features_finance đã được drop & recreate theo schema mới.")


def show_tables():
    conn = get_connection()
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        print(f"\n📊 Database: {len(tables)} tables")
        print("=" * 55)

        stages = {
            "Giá cổ phiếu (Raw)": ["raw_prices"],
            "Giá cổ phiếu (Clean)": ["clean_prices"],
            "Tin tức (Raw)": ["raw_news"],
            "Tin tức (Clean + Sentiment)": ["clean_news", "daily_news_embeddings"],
            "Tài chính (Raw)": ["raw_finance"],
            "Tài chính (Features)": ["features_finance"],
            "Merged (Input Model)": ["merged_features"],
            "Split Data (Finance)": ["train_features", "val_features", "test_features", "train_weights"],
            "Model Outputs": ["predictions", "model_metrics"],
        }

        existing_tables = {name[0] for name in tables}
        for stage, table_list in stages.items():
            stage_tables = [table for table in table_list if table in existing_tables]
            if stage_tables:
                print(f"\n{stage}:")
                for table_name in stage_tables:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    status = "✅" if count > 0 else "⬜"
                    print(f"  {status} {table_name:<30} {count:>6} rows")

        print("\n" + "=" * 55)
    finally:
        conn.close()


if __name__ == "__main__":
    create_all_tables()
    show_tables()
