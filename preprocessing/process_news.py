"""
Tiền xử lý tin tức + Sentiment Analysis bằng PhoBERT.
═════════════════════════════════════════════════════
Phụ trách: Thành viên C
Branch: feature/process-news
Chạy: python -m preprocessing.process_news

Input:  raw_news table
Output: clean_news table (có sentiment scores)
"""
import pandas as pd
import torch
from loguru import logger
from database.connection import read_table, write_table
from config.settings import SENTIMENT_MODEL, DEVICE

logger.add("logs/process_news.log", rotation="1 week")


def process_news():
    """
    Đọc raw_news → phân tích sentiment bằng PhoBERT → lưu clean_news.

    Sentiment output cho mỗi tin:
    - sentiment_score: -1 (tiêu cực) đến +1 (tích cực)
    - sentiment_pos: probability positive
    - sentiment_neg: probability negative
    - sentiment_neu: probability neutral
    """
    logger.info("Tiền xử lý tin tức + sentiment analysis...")

    df = read_table("raw_news")
    logger.info(f"Loaded {len(df)} tin tức từ raw_news")

    # TODO: Thành viên C implement
    # Gợi ý:
    # 1. Load PhoBERT sentiment model:
    #    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    #    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    #    model = AutoModelForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    #
    # 2. Với mỗi tin, chạy sentiment analysis trên title (hoặc title + content):
    #    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    #    outputs = model(**inputs)
    #    probs = torch.softmax(outputs.logits, dim=-1)
    #    # Model output: [negative, positive, neutral]
    #
    # 3. Tính sentiment_score = positive_prob - negative_prob
    # 4. Lưu: write_table(df_clean, "clean_news")
    #
    # Lưu ý:
    # - Chạy trên GPU nếu có (model.to(DEVICE))
    # - Batch processing sẽ nhanh hơn xử lý từng tin một
    # - Xử lý text trước khi đưa vào model (loại HTML tags, ký tự đặc biệt)

    raise NotImplementedError("Thành viên C cần implement hàm này")


if __name__ == "__main__":
    process_news()
