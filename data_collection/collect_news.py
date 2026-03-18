"""
Collect news related to TCB for the configured date window.
"""
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from html.parser import HTMLParser

import requests
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


def _find_all(node: _HtmlNode, *, tag: str | None = None, class_name: str | None = None) -> list[_HtmlNode]:
    matches = []
    for child in _iter_descendants(node):
        if tag is not None and child.tag != tag:
            continue
        if class_name is not None and not _has_class(child, class_name):
            continue
        matches.append(child)
    return matches


def _find_first(node: _HtmlNode | None, *, tag: str | None = None, class_name: str | None = None) -> _HtmlNode | None:
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

    return records


def parse_vnexpress_search_page(html: str) -> list[dict]:
    """Parse VnExpress search result HTML into normalized news records."""
    root = _parse_html_fragment(html)
    records = []

    for item_node in _find_all(root, tag="article", class_name="item-news"):
        title_node = _find_first(item_node, tag="h3", class_name="title-news")
        link_node = _find_first(title_node, tag="a")
        record = _build_news_record(
            date_text=_node_text(_find_first(item_node, tag="span", class_name="date")),
            title=_node_text(link_node),
            content=_node_text(_find_first(item_node, tag="p", class_name="description")),
            url=_node_attr(link_node, "href"),
            source="vnexpress",
        )
        if record is not None:
            records.append(record)

    return records


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
    # - Need requests + parser helpers above
    # - Scrape speed should be slow (time.sleep between requests)
    # - Check unique URL to avoid duplicates

    raise NotImplementedError("Team member C needs to implement this function")


if __name__ == "__main__":
    collect_news()
