"""
Export news results to CSV files for inspection.

Run:
    python scripts/export_news_results.py

This script reads `clean_news` and `daily_news_embeddings` from the project's SQLite DB
using the project's `database.connection` helpers and writes timestamped CSVs to
`outputs/news/` with `utf-8-sig` encoding for Excel compatibility.
"""
from pathlib import Path
import datetime
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so `database` package can be imported
try:
    from database.connection import table_exists, read_table
except Exception:
    repo_root = _Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from database.connection import table_exists, read_table


OUT_DIR = Path("outputs/news")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_read(table_name: str):
    try:
        if not table_exists(table_name):
            print(f"[WARN] Table '{table_name}' not found in database.")
            return None
        df = read_table(table_name)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read table '{table_name}': {e}")
        return None


def save_df_csv(df, prefix: str, ts: str):
    out_path = OUT_DIR / f"{prefix}_{ts}.csv"
    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save {out_path}: {e}")
    return out_path


def shorten_preview(df, col='content', max_chars=300):
    if col in df.columns:
        df = df.copy()
        df[col] = df[col].astype(str).str.slice(0, max_chars)
    return df


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export clean_news
    df_clean = safe_read('clean_news')
    if df_clean is None or df_clean.empty:
        print("No clean_news table or it's empty.")
    else:
        # shorten content for CSV preview
        df_out = shorten_preview(df_clean, 'content', 500)
        save_df_csv(df_out, 'clean_news', ts)

    # Export daily_news_embeddings
    df_daily = safe_read('daily_news_embeddings')
    if df_daily is None or df_daily.empty:
        print("No daily_news_embeddings table or it's empty.")
    else:
        save_df_csv(df_daily, 'daily_news_embeddings', ts)


if __name__ == '__main__':
    main()
