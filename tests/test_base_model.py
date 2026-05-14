import unittest

import numpy as np
import pandas as pd

from tests.support import install_loguru_stub

install_loguru_stub()

from models.base_model import BasePredictor


class _DummyPredictor(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(model_name="dummy", **kwargs)

    def build_model(self, input_size: int):
        raise NotImplementedError("Model build is not needed for dataset preparation tests")


class BasePredictorDatasetTests(unittest.TestCase):
    def _build_frame(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        close = np.arange(10, 20, dtype=float)
        df = pd.DataFrame(
            {
                "date": dates,
                "feature_a": np.arange(10, dtype=float),
                "feature_b": np.arange(10, 20, dtype=float),
                "target": close,
            }
        )
        return df

    def test_prepare_time_series_data_splits_by_target_index_without_leakage(self):
        model = _DummyPredictor(lookback_days=3, train_ratio=0.6, val_ratio=0.2)

        splits = model._prepare_time_series_data(
            self._build_frame(),
            feature_cols=["feature_a", "feature_b"],
            target_col="target",
        )

        self.assertEqual(splits["train"]["target_indices"], [3, 4, 5])
        self.assertEqual(splits["val"]["target_indices"], [6, 7])
        self.assertEqual(splits["test"]["target_indices"], [8, 9])

    def test_prepare_time_series_data_tracks_boundary_dates(self):
        model = _DummyPredictor(lookback_days=3, train_ratio=0.6, val_ratio=0.2)

        splits = model._prepare_time_series_data(
            self._build_frame(),
            feature_cols=["feature_a", "feature_b"],
            target_col="target",
        )

        self.assertEqual(splits["train_end_date"], "2024-01-06")
        self.assertEqual(splits["val_end_date"], "2024-01-08")
        self.assertEqual(splits["lookback_days"], 3)


if __name__ == "__main__":
    unittest.main()
