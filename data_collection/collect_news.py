"""
Collect news related to TCB for the configured date window.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
import re
import time
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests
from requests.exceptions import HTTPError
from loguru import logger
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None
import os
from pathlib import Path

from database.connection import get_connection, write_table
from config.settings import DATA_END_DATE, DATA_START_DATE, SYMBOL


logger.add("logs/collect_news.log", rotation="1 week")


REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 1.0
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}
REQUEST_EXCEPTION = getattr(requests, "RequestException", Exception)

CAFEF_BASE_URL = "https://cafef.vn"
VNEXPRESS_SEARCH_BASE_URL = "https://timkiem.vnexpress.net/"
VNEXPRESS_QUERY = "Techcombank"
VN_TIMEZONE = timezone(timedelta(hours=7))
NEWS_COLUMNS = ["date", "title", "content", "url", "source"]


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


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(unescape(text).split())


@dataclass
class _HtmlNode:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[object] = field(default_factory=list)


class _HtmlTreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _HtmlNode("document")
        self._stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = _HtmlNode(
            tag=tag,
            attrs={key: value or "" for key, value in attrs if key},
        )
        self._stack[-1].children.append(node)
        self._stack.append(node)

    def handle_startendtag(self, tag, attrs):
        node = _HtmlNode(
            tag=tag,
            attrs={key: value or "" for key, value in attrs if key},
        )
        self._stack[-1].children.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                break

    def handle_data(self, data):
        if data:
            self._stack[-1].children.append(data)


def _parse_html_fragment(html: str) -> _HtmlNode:
    parser = _HtmlTreeBuilder()
    parser.feed(html)
    parser.close()
    return parser.root


def _has_class(node: _HtmlNode, class_name: str) -> bool:
    return class_name in node.attrs.get("class", "").split()


def _iter_descendants(node: _HtmlNode):
    for child in node.children:
        if isinstance(child, _HtmlNode):
            yield child
            yield from _iter_descendants(child)


def _find_all(
    node: _HtmlNode,
    *,
    tag: str | None = None,
    class_name: str | None = None,
) -> list[_HtmlNode]:
    matches = []
    for child in _iter_descendants(node):
        if tag is not None and child.tag != tag:
            continue
        if class_name is not None and not _has_class(child, class_name):
            continue
        matches.append(child)
    return matches


def _find_first(
    node: _HtmlNode | None,
    *,
    tag: str | None = None,
    class_name: str | None = None,
) -> _HtmlNode | None:
    if node is None:
        return None
    matches = _find_all(node, tag=tag, class_name=class_name)
    return matches[0] if matches else None


def _collect_text(node: _HtmlNode | None) -> str:
    if node is None:
        return ""

    parts = []
    for child in node.children:
        if isinstance(child, _HtmlNode):
            parts.append(_collect_text(child))
        else:
            parts.append(child)
    return "".join(parts)


def _node_text(node: _HtmlNode | None) -> str:
    return _clean_text(_collect_text(node))


def _node_attr(node: _HtmlNode | None, attr_name: str) -> str:
    if node is None:
        return ""
    return _clean_text(node.attrs.get(attr_name, ""))


def _build_news_record(
    *,
    date_text: str,
    title: str,
    content: str,
    url: str,
    source: str,
) -> dict | None:
    date = normalize_date_text(date_text)
    normalized_title = _clean_text(title)
    normalized_url = _clean_text(url)

    if not (date and normalized_title and normalized_url):
        return None

    return {
        "date": date,
        "title": normalized_title,
        "content": _clean_text(content),
        "url": normalized_url,
        "source": source,
    }


def extract_cafef_date_from_url(url: str) -> str | None:
    """Extract CafeF publication date from the numeric article slug."""
    if not url:
        return None

    path = url.split("?", maxsplit=1)[0]
    match = re.search(r"-([0-9]+)\.chn$", path)
    if match is None:
        return None

    digits = match.group(1)
    candidate = None
    if digits.startswith("188") and len(digits) >= 15:
        candidate = digits[3:15]
    elif len(digits) >= 12:
        candidate = digits[:12]

    if not candidate:
        return None

    try:
        return datetime.strptime(candidate, "%y%m%d%H%M%S").date().isoformat()
    except ValueError:
        return None


def normalize_vnexpress_timestamp(timestamp_text: str) -> str | None:
    """Convert VnExpress Unix timestamp attributes into YYYY-MM-DD."""
    if not timestamp_text:
        return None

    try:
        timestamp_value = int(float(timestamp_text))
    except (TypeError, ValueError):
        return None

    return datetime.fromtimestamp(timestamp_value, tz=VN_TIMEZONE).date().isoformat()


def parse_cafef_search_page(html: str) -> list[dict]:
    """Parse CafeF search result HTML into normalized news records."""
    root = _parse_html_fragment(html)
    records = []

    for item_node in _find_all(root, tag="div", class_name="tlitem"):
        link_node = _find_first(item_node, tag="a")
        record = _build_news_record(
            date_text=_node_text(_find_first(item_node, tag="span", class_name="time")),
            title=_node_attr(link_node, "title") or _node_text(link_node),
            content=_node_text(_find_first(item_node, tag="p", class_name="sapo")),
            url=_node_attr(link_node, "href"),
            source="cafef",
        )
        if record is not None:
            records.append(record)

    timeline_nodes = [
        node
        for node in _find_all(root, tag="div", class_name="timeline")
        if _has_class(node, "list-bytags")
    ]
    for timeline_node in timeline_nodes:
        for item_node in _find_all(timeline_node, tag="div", class_name="item"):
            link_node = _find_first(item_node, tag="a", class_name="box-category-link-title")
            href = _node_attr(link_node, "href")
            absolute_url = urljoin(CAFEF_BASE_URL, href) if href else ""
            record = _build_news_record(
                date_text=extract_cafef_date_from_url(absolute_url) or "",
                title=_node_attr(link_node, "title") or _node_text(link_node),
                content=_node_text(_find_first(item_node, tag="p", class_name="sapo")),
                url=absolute_url,
                source="cafef",
            )
            if record is not None:
                records.append(record)

    return dedupe_news_records(records)


def parse_vnexpress_search_page(html: str) -> list[dict]:
    """Parse VnExpress search result HTML into normalized news records."""
    root = _parse_html_fragment(html)
    records = []

    for item_node in _find_all(root, tag="article", class_name="item-news"):
        title_node = _find_first(item_node, tag="h3", class_name="title-news")
        link_node = _find_first(title_node, tag="a")
        url = _node_attr(item_node, "data-url") or _node_attr(link_node, "href")
        date_text = (
            _node_text(_find_first(item_node, tag="span", class_name="date"))
            or _node_text(_find_first(item_node, tag="span", class_name="time-public"))
            or normalize_vnexpress_timestamp(_node_attr(item_node, "data-publishtime"))
            or ""
        )
        record = _build_news_record(
            date_text=date_text,
            title=_node_text(link_node),
            content=_node_text(_find_first(item_node, tag="p", class_name="description")),
            url=url,
            source="vnexpress",
        )
        if record is not None:
            records.append(record)

    return records


def build_cafef_search_url(page: int = 1) -> str:
    # Use the explicit cafef search path for TCB as primary entrypoint
    if page <= 1:
        return f"{CAFEF_BASE_URL}/tim-kiem/tcb.chn"
    return f"{CAFEF_BASE_URL}/tim-kiem/trang-{page}.chn?keywords={SYMBOL}"


def build_vnexpress_search_url(page: int = 1) -> str:
    params = {
        "q": VNEXPRESS_QUERY,
        "media_type": "all",
        "fromdate": "0",
        "todate": "0",
        "latest": "",
        "cate_code": "",
        "search_f": "title,tag_list",
        "date_format": "all",
    }
    if page > 1:
        params["page"] = str(page)
    return f"{VNEXPRESS_SEARCH_BASE_URL}?{urlencode(params)}"


def parse_cafef_total_pages(html: str) -> int:
    page_numbers = [int(match) for match in re.findall(r"/tim-kiem/trang-(\d+)\.chn\?keywords=", html)]
    return max(page_numbers, default=1)


def parse_vnexpress_total_pages(html: str) -> int:
    match = re.search(r'max-page="(\d+)"', html)
    return int(match.group(1)) if match else 1


def parse_vietstock_search_page(html: str) -> list[dict]:
    """Parse finance.vietstock.vn TCB news page into normalized news records."""
    records = []
    # Prefer BeautifulSoup for robust parsing
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "lxml") if hasattr(BeautifulSoup, '__call__') else BeautifulSoup(html)

        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            if not href or not text:
                continue
            href_l = href.lower()
            # heuristics: include links that look like article/news entries
            if not any(k in href_l for k in ['/tin-tuc', 'tin-tuc-su-kien', 'news', 'article', 'detail']):
                # also include if title mentions Techcombank/TCB
                if 'tc b' not in text.lower() and 'techcombank' not in text.lower() and 'tcb' not in text.lower():
                    continue

            # Build absolute URL
            url = urljoin('https://finance.vietstock.vn', href)

            # Try to locate a nearby date
            date_text = None
            # look for <time> elements near the anchor
            time_tag = a.find_previous('time') or a.find_next('time')
            if time_tag and time_tag.get_text(strip=True):
                date_text = normalize_date_text(time_tag.get_text(strip=True))

            # fallback: search ancestor spans/small elements for date-like text
            if not date_text:
                for p in a.parents:
                    if p is None:
                        break
                    span = p.find(lambda tag: tag.name in ('span', 'small') and tag.get_text(strip=True))
                    if span:
                        candidate = normalize_date_text(span.get_text(strip=True))
                        if candidate:
                            date_text = candidate
                            break

            rec = _build_news_record(date_text=date_text or '', title=text, content='', url=url, source='vietstock')
            if rec is not None:
                records.append(rec)

        return dedupe_news_records(records)

    # Fallback to simple HTML fragment parser if BeautifulSoup not available
    root = _parse_html_fragment(html)
    for a in _find_all(root, tag='a'):
        href = _node_attr(a, 'href')
        title = _node_text(a)
        if not href or not title:
            continue
        if not any(k in href.lower() for k in ['/tin-tuc', 'news', 'article']):
            if 'techcombank' not in title.lower() and 'tcb' not in title.lower():
                continue
        url = urljoin('https://finance.vietstock.vn', href)
        rec = _build_news_record(date_text='', title=title, content='', url=url, source='vietstock')
        if rec is not None:
            records.append(rec)

    return dedupe_news_records(records)


def parse_fireant_search_page(html: str) -> list[dict]:
    """Parse Fireant dashboard page for external news links related to TCB."""
    records = []
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "lxml") if hasattr(BeautifulSoup, '__call__') else BeautifulSoup(html)
        # Fireant often embeds external news links; collect anchors that contain known news hostnames or look like article links
        for a in soup.find_all('a', href=True):
            href = a['href']
            title = a.get_text(strip=True)
            if not href or not title:
                continue
            href_l = href.lower()
            if not any(k in href_l for k in ('vnexpress.net', 'cafef.vn', 'zingnews.vn', 'vnmedia.vn', 'thesaigontimes.vn', '/news', 'article', 'tin-tuc')):
                # include anchors mentioning Techcombank
                if 'techcombank' not in title.lower() and 'tcb' not in title.lower():
                    continue

            url = urljoin('https://fireant.vn', href)

            # attempt to find a date near the anchor
            date_text = None
            time_tag = a.find_previous('time') or a.find_next('time')
            if time_tag and time_tag.get_text(strip=True):
                date_text = normalize_date_text(time_tag.get_text(strip=True))

            if not date_text:
                for p in a.parents:
                    if p is None:
                        break
                    span = p.find(lambda tag: tag.name in ('span', 'small') and tag.get_text(strip=True))
                    if span:
                        candidate = normalize_date_text(span.get_text(strip=True))
                        if candidate:
                            date_text = candidate
                            break

            rec = _build_news_record(date_text=date_text or '', title=title, content='', url=url, source='fireant')
            if rec is not None:
                records.append(rec)

        return dedupe_news_records(records)

    # Fallback
    root = _parse_html_fragment(html)
    for a in _find_all(root, tag='a'):
        href = _node_attr(a, 'href')
        title = _node_text(a)
        if not href or not title:
            continue
        if 'techcombank' not in title.lower() and 'tcb' not in title.lower():
            continue
        url = urljoin('https://fireant.vn', href)
        rec = _build_news_record(date_text='', title=title, content='', url=url, source='fireant')
        if rec is not None:
            records.append(rec)

    return dedupe_news_records(records)


def fetch_search_page(session, url: str) -> str:
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    if getattr(response, "encoding", None) in (None, "ISO-8859-1"):
        apparent_encoding = getattr(response, "apparent_encoding", None)
        if apparent_encoding:
            response.encoding = apparent_encoding

    return response.text


def collect_cafef_news(
    session,
    *,
    max_pages: int | None = None,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    records = []
    total_pages = None
    page = 1

    while True:
        if max_pages is not None and page > max_pages:
            break

        url = build_cafef_search_url(page)
        logger.info(f"Fetching CafeF page {page}: {url}")
        try:
            html = fetch_search_page(session, url)
        except HTTPError as he:
            # Try fallback query-based search URL if primary path not found
            status_code = None
            try:
                status_code = he.response.status_code  # type: ignore[attr-defined]
            except Exception:
                status_code = None

            if status_code == 404 and page == 1:
                fallback = f"{CAFEF_BASE_URL}/tim-kiem.chn?keywords={SYMBOL}"
                logger.info(f"Primary CafeF URL returned 404, retrying fallback: {fallback}")
                try:
                    html = fetch_search_page(session, fallback)
                except Exception as e:
                    logger.error(f"Fallback CafeF fetch failed: {e}")
                    break
            else:
                logger.error(f"Failed fetching CafeF URL: {he}")
                break

        if total_pages is None:
            total_pages = parse_cafef_total_pages(html)

        page_records = parse_cafef_search_page(html)
        if not page_records:
            break

        records.extend(page_records)
        if page >= total_pages or (max_pages is not None and page >= max_pages):
            break

        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        page += 1

    return dedupe_news_records(records)


def collect_vietstock_news(
    session,
    *,
    max_pages: int | None = None,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Collect news from finance.vietstock.vn TCB news page."""
    url = "https://finance.vietstock.vn/TCB/tin-tuc-su-kien.htm"
    logger.info(f"Fetching Vietstock news: {url}")
    try:
        html = fetch_search_page(session, url)
    except Exception as e:
        logger.error(f"Failed fetching vietstock: {e}")
        return []

    records = parse_vietstock_search_page(html)
    return dedupe_news_records(records)


def collect_fireant_news(
    session,
    *,
    max_pages: int | None = None,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Collect news from Fireant dashboard for TCB."""
    url = "https://fireant.vn/dashboard/symbol/TCB"
    logger.info(f"Fetching Fireant dashboard: {url}")
    try:
        html = fetch_search_page(session, url)
    except Exception as e:
        logger.error(f"Failed fetching fireant: {e}")
        return []

    records = parse_fireant_search_page(html)
    return dedupe_news_records(records)


def collect_vnexpress_news(
    session,
    *,
    max_pages: int | None = None,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    records = []
    total_pages = None
    page = 1

    while True:
        if max_pages is not None and page > max_pages:
            break

        url = build_vnexpress_search_url(page)
        logger.info(f"Fetching VnExpress page {page}: {url}")
        html = fetch_search_page(session, url)
        if total_pages is None:
            total_pages = parse_vnexpress_total_pages(html)

        page_records = parse_vnexpress_search_page(html)
        if not page_records:
            break

        records.extend(page_records)
        if page >= total_pages or (max_pages is not None and page >= max_pages):
            break

        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        page += 1

    return dedupe_news_records(records)


def get_existing_news_urls() -> set[str]:
    """Read existing raw_news URLs so we can avoid duplicate inserts."""
    conn = get_connection()
    try:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_news'"
        ).fetchone()
        if table_exists is None:
            return set()

        rows = conn.execute(
            "SELECT url FROM raw_news WHERE url IS NOT NULL AND url != ''"
        ).fetchall()
        return {row[0] for row in rows if row and row[0]}
    finally:
        conn.close()


def _prepare_new_records(records: list[dict], existing_urls: set[str]) -> list[dict]:
    filtered_records = []
    for record in dedupe_news_records(records):
        if record["url"] in existing_urls:
            continue
        if not is_within_date_range(record["date"]):
            continue
        filtered_records.append(record)

    return sorted(filtered_records, key=lambda record: (record["date"], record["source"], record["url"]))


def save_news_records(records: list[dict]) -> int:
    if not records:
        return 0

    df = pd.DataFrame(records, columns=NEWS_COLUMNS)
    write_table(df, "raw_news", if_exists="append")
    return len(df)


def _build_requests_session():
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    return session


def collect_news(
    session=None,
    *,
    max_pages_per_source: int | None = None,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
    export_path: str | None = None,
) -> int:
    """
    Scrape news about TCB/Techcombank from Vietstock, CafeF and Fireant.
    Save into raw_news table.

    Columns: date, title, content, url, source
    """
    logger.info(f"{'=' * 50}")
    logger.info(f"Collecting news for {SYMBOL} | {DATA_START_DATE} -> {DATA_END_DATE}")
    logger.info(f"{'=' * 50}")

    owns_session = session is None
    if session is None:
        session = _build_requests_session()

    try:
        source_records = []
        for source_name, collector in (
            ("vietstock", collect_vietstock_news),
            ("cafef", collect_cafef_news),
            ("fireant", collect_fireant_news),
        ):
            try:
                records = collector(
                    session,
                    max_pages=max_pages_per_source,
                    request_delay_seconds=request_delay_seconds,
                )
                source_records.extend(records)
                logger.info(f"Collected {len(records)} records from {source_name}")
            except REQUEST_EXCEPTION as exc:
                logger.error(f"Failed to collect {source_name}: {exc}")

        # Export fetched records (all sources) for debugging/inspection
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fetched_path = Path("data_collection") / f"fetched_news_{ts}.csv"
            fetched_path.parent.mkdir(parents=True, exist_ok=True)
            df_fetched = pd.DataFrame(source_records, columns=NEWS_COLUMNS)
            df_fetched.to_csv(fetched_path, index=False, encoding="utf-8-sig")
            logger.info(f"Exported fetched news to {fetched_path}")
        except Exception as e:
            logger.error(f"Failed to export fetched news: {e}")

        existing_urls = get_existing_news_urls()
        new_records = _prepare_new_records(source_records, existing_urls)
        saved_count = save_news_records(new_records)

        # Export collected new records to CSV for inspection (optional)
        try:
            if new_records:
                df_export = pd.DataFrame(new_records, columns=NEWS_COLUMNS)
                ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_path = Path("data_collection") / f"collected_news_{ts2}.csv"
                out_path = Path(export_path) if export_path else default_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # Use utf-8-sig for Excel-friendly encoding
                df_export.to_csv(out_path, index=False, encoding="utf-8-sig")
                logger.info(f"Exported collected news to {out_path}")
        except Exception as e:
            logger.error(f"Failed to export collected news: {e}")

        logger.info(
            f"Finished collecting news: fetched={len(source_records)}, "
            f"new={len(new_records)}, saved={saved_count}"
        )
        return saved_count
    finally:
        if owns_session and hasattr(session, "close"):
            session.close()


if __name__ == "__main__":
    collect_news()
