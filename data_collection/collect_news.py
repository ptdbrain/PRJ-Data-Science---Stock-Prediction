"""
Thu thập tin tức liên quan TCB (3 năm).
═══════════════════════════════════════
Phụ trách: Thành viên C
Branch: feature/collect-news
Chạy: python -m data_collection.collect_news

Output: raw_news table trong SQLite
"""
import requests
from bs4 import BeautifulSoup
from loguru import logger
from database.connection import get_connection, write_table
from config.settings import SYMBOL

logger.add("logs/collect_news.log", rotation="1 week")


def collect_news():
    """
    Scrape tin tức về TCB/Techcombank từ CafeF và VnExpress.
    Lưu vào raw_news table.

    Columns: date, title, content, url, source
    """
    logger.info(f"Thu thập tin tức {SYMBOL}...")

    # TODO: Thành viên C implement
    # Gợi ý:
    # 1. Scrape từ CafeF: https://cafef.vn/tim-kiem.chn?keywords=TCB
    # 2. Scrape từ VnExpress: https://timkiem.vnexpress.net/?q=Techcombank
    # 3. Với mỗi bài: lấy date, title, content (đoạn mô tả), url, source
    # 4. Cần xử lý pagination (nhiều trang)
    # 5. Dùng headers User-Agent để tránh bị block
    # 6. Lưu bằng write_table(df, "raw_news", if_exists='append')
    #
    # Lưu ý:
    # - Cần requests + BeautifulSoup
    # - Tốc độ scrape nên chậm (time.sleep giữa các request)
    # - Check URL unique để tránh duplicate

    raise NotImplementedError("Thành viên C cần implement hàm này")


if __name__ == "__main__":
    collect_news()
