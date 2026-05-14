"""
Cấu hình tập trung cho toàn bộ project.
═══════════════════════════════════════
Mọi file khác import từ đây, KHÔNG hardcode giá trị.

Ví dụ:
    from config.settings import DB_PATH, SYMBOL, DEVICE
"""
from pathlib import Path
from datetime import date

# Optional: torch for model training (not needed for data pipeline)
try:
    import torch
except ImportError:
    torch = None

# ============================
# Paths
# ============================
PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / "database" / "tcb.db"
MODEL_DIR = PROJECT_DIR / "models" / "saved"
LOG_DIR = PROJECT_DIR / "logs"

# ============================
# Stock
# ============================
SYMBOL = "TCB"
DATA_SOURCE = "VCI"
DATA_START_DATE = "2020-01-01"
# Luôn lấy đến hôm nay để pipeline giá khớp dữ liệu mới nhất (có thể ghi đè bằng biến môi trường nếu cần)
DATA_END_DATE = date.today().strftime("%Y-%m-%d")

# ============================
# Database Table Names
# ============================
# Raw tables (data gốc, không chỉnh sửa)
TABLE_RAW_PRICES = "raw_prices"
TABLE_RAW_FINANCE = "raw_finance"
TABLE_RAW_NEWS = "raw_news"

# Clean tables (đã tiền xử lý)
TABLE_CLEAN_PRICES = "clean_prices"
TABLE_FEATURES_FINANCE = "features_finance"
# Deprecated alias kept for backward compatibility in modules/docs not yet updated.
TABLE_CLEAN_FINANCE = TABLE_FEATURES_FINANCE
TABLE_CLEAN_NEWS = "clean_news"
TABLE_DAILY_NEWS_EMBEDDINGS = "daily_news_embeddings"

# Output tables
TABLE_MERGED_FEATURES = "merged_features"
TABLE_PREDICTIONS = "predictions"
TABLE_MODEL_METRICS = "model_metrics"

# ============================
# Model Hyperparameters
# ============================
LOOKBACK_DAYS = 60       # Số ngày quá khứ dùng để predict
FORECAST_DAYS = 1        # Số ngày muốn predict
HIDDEN_SIZE = 128        # LSTM/GRU hidden units
NUM_LAYERS = 2           # Số lớp LSTM/GRU
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
DEFAULT_MODEL_NAME = "lstm"

# Train/Val/Test split ratios (theo thời gian, KHÔNG random)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================
# Feature Columns (STATIONARY — tỷ lệ %, không phải giá tuyệt đối)
# ============================
# Log returns thay vì giá tuyệt đối
PRICE_FEATURES = ['close_ret', 'open_ret', 'high_ret', 'low_ret', 'volume_ratio']

# Tất cả đều là stationary (bounded / relative)
TECHNICAL_FEATURES = [
    'sma_10_dist', 'sma_20_dist', 'sma_50_dist',   # % distance from close
    'ema_12_dist', 'ema_26_dist',                    # % distance from close
    'rsi_14',                                         # Already bounded [0, 100]
    'macd', 'macd_signal', 'macd_hist',              # Near-stationary oscillators
    'bb_position', 'bb_width',                        # Relative (replaces bb_upper/middle/lower)
    'atr_pct',                                        # % of close (replaces atr_14)
    'obv_change',                                     # % change (replaces obv)
    'price_change', 'price_change_5d',               # Already stationary
    'volatility_pct',                                 # % of close (replaces volatility_10d)
]

# Khớp với `process_finance` (features_finance) sau bước feature engineering
FINANCE_FEATURES = [
    'roe', 'roa', 'debt_to_equity',
    'net_profit_margin', 'financial_leverage',
    'roe_yoy', 'roa_yoy', 'roe_lag4', 'roa_lag4',
    'eps',          # Earnings Per Share
    'eps_yoy',      # EPS YoY growth
]

# Sentiment features: PhoBERT (DL) + TF-IDF (classical NLP)
SENTIMENT_FEATURES = [
    'daily_sentiment',      # PhoBERT score trung bình ngày
    'news_count',
    'embedding_score_mean',
    'embedding_score_std',
    'tfidf_sentiment',      # TF-IDF / Lexicon classical NLP score
]

ALL_FEATURES = PRICE_FEATURES + TECHNICAL_FEATURES + FINANCE_FEATURES + SENTIMENT_FEATURES

# ============================
# Classification Settings
# ============================
# Ngưỡng xác suất để quyết định nhãn (>= TREND_THRESHOLD → Tăng)
TREND_THRESHOLD = 0.5

# Quarter-specific reporting lag rules for when financial data becomes available.
FINANCE_REPORT_LAG_DAYS = {
    'Q1': 30,
    'Q2': 45,
    'Q3': 30,
    'Q4': 90,
}

# ============================
# NLP / Sentiment
# ============================
SENTIMENT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"

# ============================
# Device (GPU/CPU)
# ============================
if torch is not None:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = 'cpu'
