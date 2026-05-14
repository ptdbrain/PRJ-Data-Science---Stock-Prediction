"""
Tiền xử lý tin tức + Sentiment Analysis (hai phương pháp so sánh).
════════════════════════════════════════════════════════════════════
Phương pháp 1 (DL)       : PhoBERT — cột sentiment_score, sentiment_label
Phương pháp 2 (Classical) : TF-IDF + Lexicon — cột tfidf_sentiment

Input:  raw_news table  (date, title, content, url, source)
Output: clean_news table (thêm các cột sentiment và tfidf_sentiment)

Chạy:
    python -m preprocessing.process_news
"""
import re
import json
from typing import Optional

import pandas as pd

try:
    import numpy as np
except Exception:
    np = None

import torch
from utils.logger import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# If the script is run directly, the package root may not be on sys.path.
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


# ================================================================
# Constants
# ================================================================
IDX_NEG = 0
IDX_POS = 1
IDX_NEU = 2

BATCH_SIZE = 16
MAX_TOKEN_LENGTH = 256

POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15

EMBED_POSITIVE_THRESHOLD = 0.05
EMBED_NEGATIVE_THRESHOLD = -0.05

EMBED_POSITIVE_PROTOTYPES = [
    "Kết quả kinh doanh tích cực, lợi nhuận tăng, triển vọng khả quan",
    "Ngân hàng báo cáo lợi nhuận cao, cổ phiếu tăng",
    "Tin tốt: cổ tức, tăng trưởng doanh thu và lợi nhuận",
]
EMBED_NEGATIVE_PROTOTYPES = [
    "Kết quả kinh doanh kém, thua lỗ, lợi nhuận giảm",
    "Rủi ro nợ xấu tăng, sự kiện tiêu cực ảnh hưởng cổ phiếu",
    "Tin xấu: mất khách hàng, giảm doanh thu, kiện tụng",
]

# ================================================================
# Vietnamese Finance Lexicon for TF-IDF / Lexicon sentiment
# ================================================================
_POS_WORDS = {
    "tăng", "tăng trưởng", "lợi nhuận", "cổ tức", "khả quan", "tích cực",
    "phát triển", "doanh thu", "hiệu quả", "cải thiện", "thành công",
    "tốt", "xuất sắc", "vượt", "kỷ lục", "bứt phá", "tiềm năng",
    "phục hồi", "khởi sắc", "mạnh", "hợp tác", "ký kết", "lãi",
    "thặng dư", "đạt", "hoàn thành", "chiến lược", "mở rộng",
}
_NEG_WORDS = {
    "giảm", "thua lỗ", "nợ xấu", "rủi ro", "khó khăn", "tiêu cực",
    "suy giảm", "đổ xuống", "mất", "vỡ", "phá sản", "kiện", "truy tố",
    "thiệt hại", "xuống", "thất", "rào cản", "chịu", "khủng hoảng",
    "giải thể", "thu hẹp", "cắt giảm", "lỗ", "nợ", "trì hoãn",
    "chậm", "gian lận", "vi phạm",
}

CLEAN_NEWS_COLUMNS = [
    "date", "title", "content", "url", "source",
    "sentiment_neg", "sentiment_pos", "sentiment_neu",
    "sentiment_score", "sentiment_label",
    "tfidf_sentiment",
    "embedding_score", "embedding_label",
    "daily_sentiment", "daily_tfidf_sentiment", "news_count",
]

EMBEDDING_COL = "embedding"


# ================================================================
# Helpers
# ================================================================
def _serialize_vector(v):
    """Serialize embedding-like objects to a JSON string for DB storage."""
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


# ================================================================
# Classical NLP: TF-IDF + Lexicon Sentiment
# ================================================================
def compute_tfidf_sentiment(texts: list) -> list:
    """
    Tính điểm sentiment theo phương pháp TF-IDF + Lexicon (NLP cổ điển).

    Thuật toán:
      1. Tokenise văn bản (lowercase, tách từ theo khoảng trắng).
      2. Đếm tần suất từ positive và negative trong lexicon.
      3. Score = (n_pos - n_neg) / (n_pos + n_neg + 1)  ∈ (-1, 1)

    So sánh với PhoBERT: phương pháp này không cần GPU, chạy nhanh,
    dễ giải thích — phù hợp làm baseline / feature bổ sung.
    """
    scores = []
    for text in texts:
        if not text:
            scores.append(0.0)
            continue
        tokens = text.lower().split()
        n_pos = sum(1 for t in tokens if t in _POS_WORDS)
        n_neg = sum(1 for t in tokens if t in _NEG_WORDS)
        score = (n_pos - n_neg) / (n_pos + n_neg + 1)
        scores.append(round(score, 6))
    return scores


# ================================================================
# PhoBERT Model
# ================================================================
def load_sentiment_model(model_name: str = SENTIMENT_MODEL):
    """Tải PhoBERT tokenizer + classification model."""
    logger.info(f"Đang tải model: {model_name} (device={DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    logger.info("Model đã sẵn sàng.")
    return tokenizer, model


def predict_sentiment_batch(
    texts: list,
    tokenizer,
    model,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_TOKEN_LENGTH,
) -> list:
    results = []
    total = len(texts)

    for batch_start in range(0, total, batch_size):
        batch_texts = texts[batch_start: batch_start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

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


def label_from_score(score: float) -> str:
    """
    Chuyển sentiment_score thành nhãn 3 lớp.

    score >= +0.15  → 'positive'
    score <= -0.15  → 'negative'
    otherwise       → 'neutral'
    """
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


# ================================================================
# Daily Aggregation
# ================================================================
def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp sentiment (PhoBERT + TF-IDF) theo ngày."""
    agg_dict = {"daily_sentiment": ("sentiment_score", "mean"),
                "news_count": ("sentiment_score", "count")}
    if "tfidf_sentiment" in df.columns:
        agg_dict["daily_tfidf_sentiment"] = ("tfidf_sentiment", "mean")

    daily = (
        df.groupby("date", sort=False)
        .agg(**agg_dict)
        .reset_index()
    )
    daily["daily_sentiment"] = daily["daily_sentiment"].round(6)
    if "daily_tfidf_sentiment" in daily.columns:
        daily["daily_tfidf_sentiment"] = daily["daily_tfidf_sentiment"].round(6)

    drop_cols = [c for c in ["daily_sentiment", "daily_tfidf_sentiment", "news_count"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.merge(daily, on="date", how="left")
    return df


# ================================================================
# Main Pipeline
# ================================================================
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

    # 2. Kiểm tra cột
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

    empty_mask = df["_input_text"].str.len() == 0
    if empty_mask.any():
        logger.warning(f"Bỏ qua {empty_mask.sum()} bài có nội dung rỗng.")
        df = df[~empty_mask].copy()

    # 4. Phương pháp 2 (Classical): TF-IDF + Lexicon (NHANH, không cần GPU)
    logger.info("Tính TF-IDF/Lexicon classical sentiment...")
    df["tfidf_sentiment"] = compute_tfidf_sentiment(df["_input_text"].tolist())
    logger.info("  ✅ tfidf_sentiment hoàn tất")

    # 5. Phương pháp 1 (DL): PhoBERT
    tokenizer, model = load_sentiment_model()
    logger.info(f"Chạy PhoBERT sentiment analysis trên {len(df)} bài (batch={BATCH_SIZE})...")
    texts = df["_input_text"].tolist()
    sentiment_records = predict_sentiment_batch(texts, tokenizer, model)

    sentiment_df = pd.DataFrame(sentiment_records, index=df.index)
    df = pd.concat([df, sentiment_df], axis=1)
    df.drop(columns=["_input_text"], inplace=True)

    # 6. Optional: SentenceTransformer embeddings
    if SentenceTransformer is not None:
        try:
            logger.info("Tạo embeddings cho từng bài bằng SentenceTransformer...")
            emb_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            texts_for_emb = (df.get("title", "").fillna("") + ". " + df.get("content", "").fillna(""))
            embeddings = emb_model.encode(texts_for_emb.tolist(), batch_size=32, show_progress_bar=False)
            df[EMBEDDING_COL] = [e.tolist() for e in embeddings]
            logger.info(f"Embeddings tạo xong (dim={embeddings.shape[1] if hasattr(embeddings, 'shape') else '?'})")

            # Embedding similarity score
            try:
                if np is not None and embeddings is not None:
                    emb_arr = np.asarray(embeddings, dtype=float)
                    pos_proto = np.asarray(emb_model.encode(EMBED_POSITIVE_PROTOTYPES, show_progress_bar=False), dtype=float)
                    neg_proto = np.asarray(emb_model.encode(EMBED_NEGATIVE_PROTOTYPES, show_progress_bar=False), dtype=float)
                    emb_norm = np.linalg.norm(emb_arr, axis=1)
                    pos_norm = np.linalg.norm(pos_proto, axis=1)
                    neg_norm = np.linalg.norm(neg_proto, axis=1)
                    sims_pos = (emb_arr @ pos_proto.T) / (emb_norm[:, None] * pos_norm[None, :] + 1e-12)
                    sims_neg = (emb_arr @ neg_proto.T) / (emb_norm[:, None] * neg_norm[None, :] + 1e-12)
                    scores = (sims_pos.mean(axis=1) - sims_neg.mean(axis=1)).round(6)
                    df["embedding_score"] = scores.tolist()

                    def _emb_label(s):
                        if s is None:
                            return None
                        if s >= EMBED_POSITIVE_THRESHOLD:
                            return "positive"
                        if s <= EMBED_NEGATIVE_THRESHOLD:
                            return "negative"
                        return "neutral"

                    df["embedding_label"] = [_emb_label(float(s)) if s is not None else None
                                             for s in df["embedding_score"].tolist()]
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

    # Daily embeddings aggregation
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
                        logger.debug(f"Skip group for date={date}: {e}")
                        continue
                    mean_vec = arr.mean(axis=0).round(6).tolist()
                    std_vec = arr.std(axis=0).round(6).tolist()
                    esc_mean = esc_std = None
                    try:
                        if "embedding_score" in group.columns and group["embedding_score"].notnull().any():
                            esc_mean = float(group["embedding_score"].astype(float).mean().round(6))
                            esc_std = float(group["embedding_score"].astype(float).std().round(6)) if len(group) > 1 else 0.0
                    except Exception:
                        pass
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
                    daily_df["embedding_mean"] = daily_df["embedding_mean"].apply(_serialize_vector)
                    daily_df["embedding_std"] = daily_df["embedding_std"].apply(_serialize_vector)
                    write_table(daily_df, "daily_news_embeddings", if_exists="replace")
                    logger.info(f"Saved daily_news_embeddings ({len(daily_df)} rows)")
        except Exception as e:
            logger.warning(f"Không thể tổng hợp embeddings theo ngày: {e}")

    # 7. Tổng hợp sentiment theo ngày (cả 2 phương pháp)
    logger.info("Tổng hợp sentiment theo ngày (PhoBERT + TF-IDF)...")
    df = aggregate_daily_sentiment(df)

    # 8. Lưu clean_news
    output_cols = [c for c in CLEAN_NEWS_COLUMNS if c in df.columns]
    if EMBEDDING_COL in df.columns:
        output_cols.append(EMBEDDING_COL)
    df_clean = df[output_cols].copy()
    if EMBEDDING_COL in df_clean.columns:
        df_clean[EMBEDDING_COL] = df_clean[EMBEDDING_COL].apply(_serialize_vector)

    write_table(df_clean, "clean_news", if_exists="replace")
    row_count = len(df_clean)
    logger.info(f"Sentiment analysis hoàn tất. Đã lưu {row_count} dòng vào clean_news.")
    logger.info("=" * 60)

    # 9. Tóm tắt
    label_counts = df_clean["sentiment_label"].value_counts()
    logger.info("Phân phối nhãn PhoBERT sentiment:")
    for label, count in label_counts.items():
        pct = count / row_count * 100
        logger.info(f"  {label:>10}: {count:4d} ({pct:.1f}%)")

    if "tfidf_sentiment" in df_clean.columns:
        tfidf_pos = (df_clean["tfidf_sentiment"] > 0.05).sum()
        tfidf_neg = (df_clean["tfidf_sentiment"] < -0.05).sum()
        tfidf_neu = row_count - tfidf_pos - tfidf_neg
        logger.info("Phân phối TF-IDF sentiment:")
        logger.info(f"  {'positive':>10}: {tfidf_pos:4d} ({tfidf_pos/row_count*100:.1f}%)")
        logger.info(f"  {'negative':>10}: {tfidf_neg:4d} ({tfidf_neg/row_count*100:.1f}%)")
        logger.info(f"  {'neutral':>10}: {tfidf_neu:4d} ({tfidf_neu/row_count*100:.1f}%)")

    return row_count


if __name__ == "__main__":
    process_news()
