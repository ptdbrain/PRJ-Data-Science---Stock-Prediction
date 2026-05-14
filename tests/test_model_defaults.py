import unittest
from unittest.mock import patch

import pandas as pd

from tests.support import install_loguru_stub

install_loguru_stub()

import models.predict as predict_module
import scripts.predict_lstm as predict_lstm_script
import scripts.train_lstm as train_lstm_script


class ModelDefaultTests(unittest.TestCase):
    def test_train_and_predict_cli_share_same_default_model_name(self):
        train_args = train_lstm_script.build_parser().parse_args([])
        predict_args = predict_lstm_script.build_parser().parse_args([])

        self.assertEqual(train_args.name, "lstm")
        self.assertEqual(predict_args.name, "lstm")

    def test_predict_all_loads_explicit_best_model_artifact_name(self):
        class FakeModel:
            loaded_name = None
            feature_cols = ["open"]
            lookback_days = 2
            threshold = 0.5

            def load(self, name=None):
                type(self).loaded_name = name

            def predict_proba(self, df):
                return 0.75

        metrics_df = pd.DataFrame([{"model_name": "lstm", "accuracy": 60.0, "is_best": 1}])
        merged_df = pd.DataFrame(
            [
                {"date": "2024-01-01", "open": 10.0, "target": 0},
                {"date": "2024-01-02", "open": 11.0, "target": 1},
                {"date": "2024-01-03", "open": 12.0, "target": 0},
                {"date": "2024-01-04", "open": 13.0, "target": 1},
            ]
        )

        saved = {}

        def fake_read_table(name):
            if name == "model_metrics":
                return metrics_df
            if name == "merged_features":
                return merged_df
            raise ValueError(name)

        with patch.dict(predict_module.DEEP_MODEL_MAP, {"lstm": FakeModel}, clear=True), patch.object(
            predict_module, "read_table", side_effect=fake_read_table
        ), patch.object(
            predict_module,
            "write_table",
            side_effect=lambda df, name, **kwargs: saved.setdefault(name, df.copy()),
        ):
            predict_module.predict_all()

        self.assertEqual(saved["predictions"]["model_name"].iloc[0], "lstm")
        self.assertEqual(FakeModel.loaded_name, "lstm")
        self.assertEqual(saved["predictions"]["date"].tolist(), ["2024-01-03", "2024-01-04"])
        self.assertEqual(saved["predictions"]["actual_trend"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
