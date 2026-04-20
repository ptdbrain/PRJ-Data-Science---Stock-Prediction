"""
Export per-article single score CSV.

This script reads `clean_news` and writes a CSV containing a single column
`score`. For each article we prefer `embedding_score` (if present); otherwise
we fall back to `sentiment_score`.

Run from project root:
    python scripts/export_news_scores.py
"""
from pathlib import Path
import datetime
import sys
import pandas as pd

# Ensure repository root on sys.path so `database` package imports work
try:
    from database.connection import table_exists, read_table
except Exception:
    from pathlib import Path as _Path
    repo_root = _Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from database.connection import table_exists, read_table


OUT_DIR = Path("outputs/news")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_read(table_name: str):
    if not table_exists(table_name):
        print(f"[WARN] Table '{table_name}' not found")
        return None
    return read_table(table_name)


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = safe_read("clean_news")
    if df is None or df.empty:
        print("No clean_news available to export scores.")
        return

    # Prefer embedding_score if available per-row, else fallback to sentiment_score
    if "embedding_score" in df.columns:
        emb = pd.to_numeric(df["embedding_score"], errors="coerce")
    else:
        emb = pd.Series([pd.NA] * len(df))

    if "sentiment_score" in df.columns:
        sent = pd.to_numeric(df["sentiment_score"], errors="coerce")
    else:
        sent = pd.Series([pd.NA] * len(df))

    scores = emb.fillna(sent).astype(float)

    # Min-max scale to [0,1]
    valid = scores.dropna()
    if valid.empty:
        print("No numeric scores available to export.")
        return

    min_v = float(valid.min())
    max_v = float(valid.max())
    if min_v == max_v:
        scaled = scores.apply(lambda _: 0.5)
    else:
        scaled = (scores - min_v) / (max_v - min_v)
        scaled = scaled.clip(0.0, 1.0)

    out_df = pd.DataFrame({"score_raw": scores, "score": scaled})
    out_path = OUT_DIR / f"article_scores_{ts}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved single-score CSV: {out_path}")


if __name__ == "__main__":
    main()
