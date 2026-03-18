import sys
import types
import unittest


if "bs4" not in sys.modules:
    bs4_module = types.ModuleType("bs4")
    bs4_module.BeautifulSoup = object
    sys.modules["bs4"] = bs4_module

if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")

if "loguru" not in sys.modules:
    loguru_module = types.ModuleType("loguru")

    class _Logger:
        def add(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    loguru_module.logger = _Logger()
    sys.modules["loguru"] = loguru_module

if "database" not in sys.modules:
    sys.modules["database"] = types.ModuleType("database")

if "database.connection" not in sys.modules:
    database_connection_module = types.ModuleType("database.connection")
    database_connection_module.get_connection = lambda *args, **kwargs: None
    database_connection_module.write_table = lambda *args, **kwargs: None
    sys.modules["database.connection"] = database_connection_module

if "config" not in sys.modules:
    sys.modules["config"] = types.ModuleType("config")

if "config.settings" not in sys.modules:
    settings_module = types.ModuleType("config.settings")
    settings_module.SYMBOL = "TCB"
    settings_module.DATA_START_DATE = "2022-01-01"
    settings_module.DATA_END_DATE = "2025-03-12"
    sys.modules["config.settings"] = settings_module

from data_collection.collect_news import (
    dedupe_news_records,
    is_within_date_range,
    normalize_date_text,
)


class CollectNewsHelperTests(unittest.TestCase):
    def test_normalize_date_text_keeps_iso_date(self):
        self.assertEqual(normalize_date_text("2024-11-05"), "2024-11-05")

    def test_normalize_date_text_converts_dmy_date(self):
        self.assertEqual(normalize_date_text("05/11/2024"), "2024-11-05")

    def test_is_within_date_range_accepts_in_range_date(self):
        self.assertTrue(is_within_date_range("2024-11-05"))

    def test_is_within_date_range_rejects_out_of_range_date(self):
        self.assertFalse(is_within_date_range("2021-12-31"))

    def test_dedupe_news_records_keeps_latest_record_for_duplicate_url(self):
        records = [
            {"url": "https://example.com/a", "title": "old"},
            {"url": "https://example.com/b", "title": "keep"},
            {"url": "https://example.com/a", "title": "new"},
        ]

        self.assertEqual(
            dedupe_news_records(records),
            [
                {"url": "https://example.com/b", "title": "keep"},
                {"url": "https://example.com/a", "title": "new"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
