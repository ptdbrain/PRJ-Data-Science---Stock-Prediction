"""
Tiền xử lý tin tức + Sentiment Analysis bằng PhoBERT.
═════════════════════════════════════════════════════
Input:  raw_news table  (date, title, content, url, source)
Output: clean_news table (thêm cột sentiment_neg, sentiment_pos,
                           sentiment_neu, sentiment_score,
                           sentiment_label, daily_sentiment, news_count)

Chạy:
    python -m preprocessing.process_news
"""
import re
from typing import Optional

import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import DEVICE, SENTIMENT_MODEL
from database.connection import read_table, write_table

logger.add("logs/process_news.log", rotation="1 week")


IDX_NEG = 0
IDX_POS = 1
IDX_NEU = 2

BATCH_SIZE = 16          
MAX_TOKEN_LENGTH = 256 

POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15

CLEAN_NEWS_COLUMNS = [
    "date", "title", "content", "url", "source",
    "sentiment_neg", "sentiment_pos", "sentiment_neu",
    "sentiment_score", "sentiment_label",
    "daily_sentiment", "news_count",
]

_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_SPECIAL = re.compile(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:()\-]")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_SPECIAL.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def build_input_text(title: str, content: str, max_content_chars: int = 200) -> str:
    title_clean = clean_text(title)
    content_clean = clean_text(content)
    if content_clean:
        content_preview = content_clean[:max_content_chars]
        return f"{title_clean}. {content_preview}"
    return title_clean


# Model
def load_sentiment_model(model_name: str = SENTIMENT_MODEL):
    """
    Tải PhoBERT tokenizer + classification model.
    Trả về (tokenizer, model) đã được đưa lên DEVICE.
    """
    logger.info(f"Đang tải model: {model_name} (device={DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    logger.info("Model đã sẵn sàng.")
    return tokenizer, model



# Inference
def predict_sentiment_batch(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_TOKEN_LENGTH,
) -> list[dict]:

    results = []
    total = len(texts)

    for batch_start in range(0, total, batch_size):
        batch_texts = texts[batch_start: batch_start + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        # Inference (no gradient)
        with torch.no_grad():
            logits = model(**encoded).logits  # (batch, 3)

        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        for prob in probs:
            neg = prob[IDX_NEG]
            pos = prob[IDX_POS]
            neu = prob[IDX_NEU]
            score = pos - neg
            results.append(
                {
                    "sentiment_neg": round(neg, 6),
                    "sentiment_pos": round(pos, 6),
                    "sentiment_neu": round(neu, 6),
                    "sentiment_score": round(score, 6),
                    "sentiment_label": label_from_score(score),
                }
            )

        logger.debug(
            f"Processed batch {batch_start // batch_size + 1}/"
            f"{(total - 1) // batch_size + 1}"
        )

    return results

# Labelling
def label_from_score(score: float) -> str:
    """
    Chuyển sentiment_score thành nhãn 3 lớp.

    score >= +0.15  → 'positive'  (tích cực)
    score <= -0.15  → 'negative'  (tiêu cực)
    otherwise       → 'neutral'   (trung hòa)
    """
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"



# Daily Aggregation
def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("date", sort=False)["sentiment_score"]
        .agg(daily_sentiment="mean", news_count="count")
        .reset_index()
    )
    daily["daily_sentiment"] = daily["daily_sentiment"].round(6)

    df = df.drop(columns=["daily_sentiment", "news_count"], errors="ignore")
    df = df.merge(daily, on="date", how="left")
    return df

# Main Pipeline


def process_news() -> int:
    logger.info("=" * 60)
    logger.info("Bắt đầu tiền xử lý tin tức + sentiment analysis")
    logger.info("=" * 60)

    # 1. Đọc dữ liệu thô
    df = read_table("raw_news")
    logger.info(f"Loaded {len(df)} tin tức từ raw_news")

    if df.empty:
        logger.warning("raw_news rỗng — không có gì để xử lý.")
        return 0

    # 2. Đảm bảo các cột cần thiết tồn tại
    for col in ("title", "content", "date"):
        if col not in df.columns:
            raise ValueError(f"Thiếu cột '{col}' trong raw_news")

    # 3. Làm sạch và ghép văn bản đầu vào
    logger.info("Làm sạch văn bản...")
    df["_input_text"] = df.apply(
        lambda row: build_input_text(
            row.get("title", "") or "",
            row.get("content", "") or "",
        ),
        axis=1,
    )

    # Loại bỏ các bài có text rỗng sau khi làm sạch
    empty_mask = df["_input_text"].str.len() == 0
    if empty_mask.any():
        logger.warning(f"Bỏ qua {empty_mask.sum()} bài có nội dung rỗng.")
        df = df[~empty_mask].copy()

    # 4. Load model
    tokenizer, model = load_sentiment_model()

    # 5. Batch inference
    logger.info(f"Chạy sentiment analysis trên {len(df)} bài (batch={BATCH_SIZE})...")
    texts = df["_input_text"].tolist()
    sentiment_records = predict_sentiment_batch(texts, tokenizer, model)

    sentiment_df = pd.DataFrame(sentiment_records, index=df.index)
    df = pd.concat([df, sentiment_df], axis=1)
    df.drop(columns=["_input_text"], inplace=True)

    # 6. Tổng hợp sentiment theo ngày
    logger.info("Tổng hợp sentiment theo ngày...")
    df = aggregate_daily_sentiment(df)

    # 7. Lưu clean_news
    output_cols = [c for c in CLEAN_NEWS_COLUMNS if c in df.columns]
    df_clean = df[output_cols].copy()

    write_table(df_clean, "clean_news", if_exists="replace")
    row_count = len(df_clean)
    logger.info(f"Sentiment analysis hoàn tất. Đã lưu {row_count} dòng vào clean_news.")
    logger.info("=" * 60)

    # 8. In tóm tắt phân phối nhãn
    label_counts = df_clean["sentiment_label"].value_counts()
    logger.info("Phân phối nhãn sentiment:")
    for label, count in label_counts.items():
        pct = count / row_count * 100
        logger.info(f"  {label:>10}: {count:4d} ({pct:.1f}%)")

    return row_count


if __name__ == "__main__":
    process_news()
