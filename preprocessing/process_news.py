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
import json
try:
    import numpy as np
except Exception:
    np = None
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# If the script is run directly, the package root may not be on sys.path.
# Try importing normally; if it fails, add the repository root to sys.path
# so `config` can be imported when running `python preprocessing/process_news.py`.
try:
    from config.settings import DEVICE, SENTIMENT_MODEL
except Exception:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from config.settings import DEVICE, SENTIMENT_MODEL
from database.connection import read_table, write_table

logger.add("logs/process_news.log", rotation="1 week")

# Optional sentence embeddings (SentenceTransformers). If not installed, skip.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


IDX_NEG = 0
IDX_POS = 1
IDX_NEU = 2

BATCH_SIZE = 16          
MAX_TOKEN_LENGTH = 256 

POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15

# Embedding-based scoring thresholds (prototype similarity)
EMBED_POSITIVE_THRESHOLD = 0.05
EMBED_NEGATIVE_THRESHOLD = -0.05

# Prototype sentences (Vietnamese, finance domain) for quick similarity scoring
EMBED_POSITIVE_PROTOTYPES = [
    "Kết quả kinh doanh tích cực, lợi nhuận tăng, triển vọng khả quan",
    "Ngân hàng báo cáo lợi nhuận cao, cổ phiếu tăng",
    "Tin tốt: cổ tức, tăng trưởng doanh thu và lợi nhuận"
]
EMBED_NEGATIVE_PROTOTYPES = [
    "Kết quả kinh doanh kém, thua lỗ, lợi nhuận giảm",
    "Rủi ro nợ xấu tăng, sự kiện tiêu cực ảnh hưởng cổ phiếu",
    "Tin xấu: mất khách hàng, giảm doanh thu, kiện tụng"
]

CLEAN_NEWS_COLUMNS = [
    "date", "title", "content", "url", "source",
    "sentiment_neg", "sentiment_pos", "sentiment_neu",
    "sentiment_score", "sentiment_label",
    "embedding_score", "embedding_label",
    "daily_sentiment", "news_count",
]

# Per-article embedding column name
EMBEDDING_COL = "embedding"

def _serialize_vector(v):
    """Serialize embedding-like objects to a JSON string for DB storage.

    Returns None when value cannot be serialized or is None.
    """
    if v is None:
        return None
    try:
        return json.dumps(v)
    except TypeError:
        try:
            return json.dumps(list(v))
        except Exception:
            return None

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

    # Optional: compute sentence embeddings per article and store as list
    if SentenceTransformer is not None:
        try:
            logger.info("Tạo embeddings cho từng bài bằng SentenceTransformer...")
            emb_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            texts_for_emb = (df.get("title", "").fillna("") + ". " + df.get("content", "").fillna(""))
            embeddings = emb_model.encode(texts_for_emb.tolist(), batch_size=32, show_progress_bar=False)
            # store as Python lists so they can be serialized to DB/CSV
            df[EMBEDDING_COL] = [e.tolist() for e in embeddings]
            try:
                dim = embeddings.shape[1]
            except Exception:
                dim = None
            logger.info(f"Embeddings tạo xong (dim={dim})")
            # Compute embedding-based similarity score using prototypes
            try:
                if np is not None and embeddings is not None:
                    emb_arr = np.asarray(embeddings, dtype=float)
                    # encode prototypes
                    pos_proto = np.asarray(emb_model.encode(EMBED_POSITIVE_PROTOTYPES, show_progress_bar=False), dtype=float)
                    neg_proto = np.asarray(emb_model.encode(EMBED_NEGATIVE_PROTOTYPES, show_progress_bar=False), dtype=float)

                    # norms
                    emb_norm = np.linalg.norm(emb_arr, axis=1)
                    pos_norm = np.linalg.norm(pos_proto, axis=1)
                    neg_norm = np.linalg.norm(neg_proto, axis=1)

                    # cosine similarities: (n_articles, n_protos)
                    sims_pos = (emb_arr @ pos_proto.T) / (emb_norm[:, None] * pos_norm[None, :] + 1e-12)
                    sims_neg = (emb_arr @ neg_proto.T) / (emb_norm[:, None] * neg_norm[None, :] + 1e-12)

                    sim_pos_mean = sims_pos.mean(axis=1)
                    sim_neg_mean = sims_neg.mean(axis=1)
                    scores = (sim_pos_mean - sim_neg_mean).round(6)

                    df["embedding_score"] = scores.tolist()
                    # map to label using simple thresholds
                    def _emb_label(s):
                        if s is None:
                            return None
                        if s >= EMBED_POSITIVE_THRESHOLD:
                            return "positive"
                        if s <= EMBED_NEGATIVE_THRESHOLD:
                            return "negative"
                        return "neutral"

                    df["embedding_label"] = [_emb_label(float(s)) if s is not None else None for s in df["embedding_score"].tolist()]
                else:
                    df["embedding_score"] = None
                    df["embedding_label"] = None
            except Exception as e:
                logger.warning(f"Không thể tính embedding_score: {e}")
                df["embedding_score"] = None
                df["embedding_label"] = None
        except Exception as e:
            logger.warning(f"Không thể tạo embeddings: {e}")
            df[EMBEDDING_COL] = None
    else:
        logger.info("SentenceTransformer không được cài đặt; bỏ qua bước tạo embeddings.")

    # Daily aggregated embeddings (mean & std) per date
    if EMBEDDING_COL in df.columns and SentenceTransformer is not None and np is not None:
        try:
            logger.info("Tổng hợp embeddings theo ngày (mean/std)...")
            emb_rows = df[df[EMBEDDING_COL].notnull()].copy()
            if not emb_rows.empty:
                daily_records = []
                for date, group in emb_rows.groupby("date", sort=False):
                    try:
                        arr = np.vstack([np.asarray(x, dtype=float) for x in group[EMBEDDING_COL].tolist()])
                    except Exception as e:
                        logger.debug(f"Skip group for date={date} due to error: {e}")
                        continue
                    mean_vec = arr.mean(axis=0).round(6).tolist()
                    std_vec = arr.std(axis=0).round(6).tolist()
                    # compute embedding_score stats if available
                    try:
                        if "embedding_score" in group.columns and group["embedding_score"].notnull().any():
                            esc_mean = float(group["embedding_score"].astype(float).mean().round(6))
                            esc_std = float(group["embedding_score"].astype(float).std().round(6)) if len(group) > 1 else 0.0
                        else:
                            esc_mean = None
                            esc_std = None
                    except Exception:
                        esc_mean = None
                        esc_std = None

                    daily_records.append({
                        "date": date,
                        "embedding_mean": mean_vec,
                        "embedding_std": std_vec,
                        "embedding_score_mean": esc_mean,
                        "embedding_score_std": esc_std,
                        "news_count": len(group),
                    })
                if daily_records:
                    daily_df = pd.DataFrame(daily_records)
                    # serialize embedding vectors to JSON strings so SQLite can store them
                    if "embedding_mean" in daily_df.columns:
                        daily_df["embedding_mean"] = daily_df["embedding_mean"].apply(_serialize_vector)
                    if "embedding_std" in daily_df.columns:
                        daily_df["embedding_std"] = daily_df["embedding_std"].apply(_serialize_vector)
                    write_table(daily_df, "daily_news_embeddings", if_exists="replace")
                    logger.info(f"Saved daily_news_embeddings ({len(daily_df)} rows)")
                else:
                    logger.info("No per-article embeddings available for daily aggregation.")
            else:
                logger.info("No per-article embeddings available for daily aggregation.")
        except Exception as e:
            logger.warning(f"Không thể tổng hợp embeddings theo ngày: {e}")
    else:
        logger.debug("Skipping daily embeddings aggregation (missing embeddings or dependencies).")

    # 6. Tổng hợp sentiment theo ngày
    logger.info("Tổng hợp sentiment theo ngày...")
    df = aggregate_daily_sentiment(df)

    # 7. Lưu clean_news
    output_cols = [c for c in CLEAN_NEWS_COLUMNS if c in df.columns]
    # include embedding column if present
    if EMBEDDING_COL in df.columns:
        output_cols.append(EMBEDDING_COL)
    df_clean = df[output_cols].copy()
    # Serialize per-article embedding column to JSON string for DB storage
    if EMBEDDING_COL in df_clean.columns:
        df_clean[EMBEDDING_COL] = df_clean[EMBEDDING_COL].apply(_serialize_vector)

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
