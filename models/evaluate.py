"""
So sánh kết quả các models.
════════════════════════════
Chạy: python -m models.evaluate

In bảng so sánh metrics + vẽ biểu đồ.
"""
import pandas as pd
from loguru import logger
from database.connection import read_table

logger.add("logs/evaluate.log", rotation="1 week")


def compare_models():
    """In bảng so sánh tất cả models."""
    metrics = read_table("model_metrics")

    if metrics.empty:
        print("❌ Chưa có metrics. Chạy 'python -m models.train' trước.")
        return

    print("\n" + "=" * 75)
    print("📊 SO SÁNH MODELS — DỰ ĐOÁN GIÁ TCB")
    print("=" * 75)
    print(f"{'Model':<15} {'RMSE (VND)':>12} {'MAE (VND)':>12} "
          f"{'MAPE (%)':>10} {'Dir.Acc (%)':>12} {'Best':>6}")
    print("-" * 75)

    for _, row in metrics.sort_values('mape').iterrows():
        tag = "⭐" if row.get('is_best', 0) else ""
        print(f"{row['model_name']:<15} "
              f"{row['rmse']:>12,.0f} "
              f"{row['mae']:>12,.0f} "
              f"{row['mape']:>10.2f} "
              f"{row['directional_accuracy']:>12.1f} "
              f"{tag:>6}")

    print("=" * 75)


if __name__ == "__main__":
    compare_models()
