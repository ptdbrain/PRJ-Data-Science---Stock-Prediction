"""
Cấu hình tập trung cho toàn bộ project.
═══════════════════════════════════════
Mọi file khác import từ đây, KHÔNG hardcode giá trị.

Ví dụ:
    from config.settings import DB_PATH, SYMBOL, DEVICE
"""
from pathlib import Path
import torch

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
DATA_START_DATE = "2022-01-01"   # 3 năm gần nhất
DATA_END_DATE = "2025-03-12"     # Ngày cố định cho Phase 1

# ============================
# Database Table Names
# ============================
# Raw tables (data gốc, không chỉnh sửa)
TABLE_RAW_PRICES = "raw_prices"
TABLE_RAW_FINANCE = "raw_finance"
TABLE_RAW_NEWS = "raw_news"

# Clean tables (đã tiền xử lý)
TABLE_CLEAN_PRICES = "clean_prices"
TABLE_CLEAN_FINANCE = "clean_finance"
TABLE_CLEAN_NEWS = "clean_news"

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

# Train/Val/Test split ratios (theo thời gian, KHÔNG random)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================
# Feature Columns
# ============================
PRICE_FEATURES = ['open', 'high', 'low', 'close', 'volume']

TECHNICAL_FEATURES = [
    'sma_10', 'sma_20', 'sma_50',
    'ema_12', 'ema_26',
    'rsi_14',
    'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'obv',
    'price_change', 'price_change_5d',
    'volatility_10d', 'volume_sma_10'
]

FINANCE_FEATURES = [
    'roe', 'roa', 'nim',
    'pe_ratio', 'pb_ratio',
    'debt_to_equity', 'revenue_growth', 'profit_growth'
]

SENTIMENT_FEATURES = ['daily_sentiment', 'news_count']

ALL_FEATURES = PRICE_FEATURES + TECHNICAL_FEATURES + FINANCE_FEATURES + SENTIMENT_FEATURES

# ============================
# NLP / Sentiment
# ============================
SENTIMENT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"

# ============================
# Device (GPU/CPU)
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
