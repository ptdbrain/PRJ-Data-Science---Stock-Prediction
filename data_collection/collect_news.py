"""
Collect news related to TCB for the configured date window.
"""
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from loguru import logger
from database.connection import get_connection, write_table
from config.settings import DATA_START_DATE, DATA_END_DATE, SYMBOL

logger.add("logs/collect_news.log", rotation="1 week")


def normalize_date_text(date_text: str) -> str | None:
    """Normalize common date strings to YYYY-MM-DD."""
    if not date_text:
        return None

    text = date_text.strip()
    if not text:
        return None

    candidate_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]

    for fmt in candidate_formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def is_within_date_range(date_str: str) -> bool:
    """Return True when a date falls within the configured range."""
    normalized_date = normalize_date_text(date_str)
    if normalized_date is None:
        return False

    start_date = normalize_date_text(DATA_START_DATE)
    end_date = normalize_date_text(DATA_END_DATE)
    if start_date is None or end_date is None:
        raise ValueError("Invalid DATA_START_DATE or DATA_END_DATE configuration")

    return start_date <= normalized_date <= end_date


def dedupe_news_records(records: list[dict]) -> list[dict]:
    """Deduplicate records by URL while keeping the last seen record."""
    deduped_by_url = {}
    ordered_keys = []

    for index, record in enumerate(reversed(records)):
        url = record.get("url")
        key = url if url else f"__missing_url_{index}"
        if key in deduped_by_url:
            continue
        deduped_by_url[key] = record
        ordered_keys.append(key)

    return [deduped_by_url[key] for key in reversed(ordered_keys)]


def collect_news():
    """
    Scrape news about TCB/Techcombank from CafeF and VnExpress.
    Save into raw_news table.

    Columns: date, title, content, url, source
    """
    logger.info(f"Collecting news for {SYMBOL}...")

    # TODO: Team member C implement
    # Suggestions:
    # 1. Scrape from CafeF: https://cafef.vn/tim-kiem.chn?keywords=TCB
    # 2. Scrape from VnExpress: https://timkiem.vnexpress.net/?q=Techcombank
    # 3. For each article: fetch date, title, content (summary), url, source
    # 4. Handle pagination (multiple pages)
    # 5. Use User-Agent headers to avoid blocking
    # 6. Save with write_table(df, "raw_news", if_exists='append')
    #
    # Notes:
    # - Need requests + BeautifulSoup
    # - Scrape speed should be slow (time.sleep between requests)
    # - Check unique URL to avoid duplicates

    raise NotImplementedError("Team member C needs to implement this function")


if __name__ == "__main__":
    collect_news()
