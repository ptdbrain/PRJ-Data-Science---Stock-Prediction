import unittest
from unittest.mock import patch

import pandas as pd

from tests.support import install_loguru_stub

install_loguru_stub()

import preprocessing.merge_features as merge_features_module


def _build_clean_prices():
    rows = []
    dates = pd.date_range("2024-04-29", periods=4, freq="D")
    for idx, date in enumerate(dates):
        base = 100.0 + idx
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": base,
                "high": base + 1,
                "low": base - 1,
                "close": base,
                "volume": 1000 + idx,
                "sma_10": base,
                "sma_20": base,
                "sma_50": base,
                "ema_12": base,
                "ema_26": base,
                "rsi_14": 55.0,
                "macd": 0.1,
                "macd_signal": 0.1,
                "macd_hist": 0.0,
                "bb_upper": base + 2,
                "bb_middle": base,
                "bb_lower": base - 2,
                "atr_14": 1.0,
                "obv": 10_000.0,
                "price_change": 0.01,
                "price_change_5d": 0.02,
                "volatility_10d": 0.03,
                "volume_sma_10": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _build_clean_prices_with_closes(closes):
    rows = []
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    for idx, (date, close) in enumerate(zip(dates, closes)):
        base = float(close)
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": base,
                "high": base + 1,
                "low": base - 1,
                "close": base,
                "volume": 1000 + idx,
                "sma_10": base,
                "sma_20": base,
                "sma_50": base,
                "ema_12": base,
                "ema_26": base,
                "rsi_14": 55.0,
                "macd": 0.1,
                "macd_signal": 0.1,
                "macd_hist": 0.0,
                "bb_upper": base + 2,
                "bb_middle": base,
                "bb_lower": base - 2,
                "atr_14": 1.0,
                "obv": 10_000.0,
                "price_change": 0.01,
                "price_change_5d": 0.02,
                "volatility_10d": 0.03,
                "volume_sma_10": 1000.0,
            }
        )
    return pd.DataFrame(rows)


class MergeFeaturesTests(unittest.TestCase):
    def test_merge_features_applies_finance_only_after_effective_date(self):
        finance = pd.DataFrame(
            [
                {
                    "date": "2024-Q1",
                    "period_end_date": "2024-03-31",
                    "effective_date": "2024-04-30",
                    "roe": 10.0,
                    "roa": 1.0,
                    "debt_to_equity": 1.0,
                    "net_profit_margin": 20.0,
                    "financial_leverage": 8.0,
                    "roe_yoy": 1.0,
                    "roa_yoy": 1.0,
                    "roe_lag4": 9.0,
                    "roa_lag4": 0.9,
                }
            ]
        )
        news = pd.DataFrame(columns=["date", "sentiment_score", "embedding_score"])
        saved = {}

        def fake_read_table(name):
            if name == "clean_prices":
                return _build_clean_prices()
            if name == "features_finance":
                return finance
            if name == "clean_news":
                return news
            raise ValueError(name)

        with patch.object(merge_features_module, "TARGET_HORIZON_DAYS", 1), patch.object(
            merge_features_module, "MIN_TARGET_RETURN", 0.0
        ), patch.object(merge_features_module, "read_table", side_effect=fake_read_table), patch.object(
            merge_features_module,
            "write_table",
            side_effect=lambda df, name, **kwargs: saved.setdefault(name, df.copy()),
        ):
            merged = merge_features_module.merge_features()

        self.assertNotIn("2024-04-29", merged["date"].tolist())
        self.assertEqual(merged.loc[merged["date"] == "2024-04-30", "roe"].iloc[0], 10.0)

    def test_merge_features_does_not_fallback_to_raw_finance(self):
        read_calls = []
        news = pd.DataFrame(columns=["date", "sentiment_score", "embedding_score"])

        def fake_read_table(name):
            read_calls.append(name)
            if name == "clean_prices":
                return _build_clean_prices()
            if name == "features_finance":
                raise ValueError("features_finance missing")
            if name == "clean_news":
                return news
            if name == "raw_finance":
                return pd.DataFrame([{"date": "2024-Q1", "roe": 999.0}])
            raise ValueError(name)

        with patch.object(merge_features_module, "TARGET_HORIZON_DAYS", 1), patch.object(
            merge_features_module, "MIN_TARGET_RETURN", 0.0
        ), patch.object(merge_features_module, "read_table", side_effect=fake_read_table), patch.object(
            merge_features_module, "write_table"
        ):
            merged = merge_features_module.merge_features()

        self.assertNotIn("raw_finance", read_calls)
        self.assertNotIn("roe", merged.columns)

    def test_merge_features_drops_last_row_without_future_close(self):
        news = pd.DataFrame(columns=["date", "sentiment_score", "embedding_score"])

        def fake_read_table(name):
            if name == "clean_prices":
                return _build_clean_prices()
            if name == "features_finance":
                raise ValueError("features_finance missing")
            if name == "clean_news":
                return news
            raise ValueError(name)

        with patch.object(merge_features_module, "TARGET_HORIZON_DAYS", 1), patch.object(
            merge_features_module, "MIN_TARGET_RETURN", 0.0
        ), patch.object(merge_features_module, "read_table", side_effect=fake_read_table), patch.object(
            merge_features_module, "write_table"
        ):
            merged = merge_features_module.merge_features()

        self.assertEqual(merged["date"].tolist(), ["2024-04-29", "2024-04-30", "2024-05-01"])
        self.assertTrue(merged["next_close"].notna().all())
        self.assertEqual(merged["target"].tolist(), [1, 1, 1])

    def test_merge_features_builds_configured_horizon_target_and_return(self):
        news = pd.DataFrame(columns=["date", "sentiment_score", "embedding_score"])

        def fake_read_table(name):
            if name == "clean_prices":
                return _build_clean_prices_with_closes([100, 101, 102, 103, 104, 106, 105])
            if name == "features_finance":
                raise ValueError("features_finance missing")
            if name == "clean_news":
                return news
            raise ValueError(name)

        with patch.object(merge_features_module, "TARGET_HORIZON_DAYS", 5), patch.object(
            merge_features_module, "MIN_TARGET_RETURN", 0.01
        ), patch.object(merge_features_module, "read_table", side_effect=fake_read_table), patch.object(
            merge_features_module, "write_table"
        ):
            merged = merge_features_module.merge_features()

        self.assertEqual(merged["date"].tolist(), ["2024-01-01", "2024-01-02"])
        self.assertEqual(merged["next_close"].tolist(), [106.0, 105.0])
        self.assertEqual(merged["forward_return"].round(4).tolist(), [0.06, 0.0396])
        self.assertEqual(merged["target"].tolist(), [1, 1])
        self.assertEqual(merged["target_1d"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
