"""
So sánh kết quả models + Backtest chiến lược giao dịch.
══════════════════════════════════════════════════════════
Chạy: python -m models.evaluate

Hai chức năng:
  1. compare_models()  — in bảng classification metrics
  2. backtest()        — mô phỏng chiến lược và so sánh với Buy-and-Hold
"""
import numpy as np
import pandas as pd
from utils.logger import logger
from database.connection import read_table
from config.settings import TRAIN_RATIO, VAL_RATIO

logger.add("logs/evaluate.log", rotation="1 week")


# ================================================================
# 1. Model Comparison Table
# ================================================================
def compare_models():
    """In bảng so sánh classification metrics của tất cả models."""
    metrics = read_table("model_metrics")

    if metrics.empty:
        print("❌ Chưa có metrics. Chạy 'python -m models.train' trước.")
        return

    print("\n" + "=" * 85)
    print("📊 SO SÁNH MODELS — DỰ ĐOÁN XU HƯỚNG TCB (Classification)")
    print("=" * 85)
    print(f"{'Model':<25} {'Acc (%)':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Best':>6}")
    print("-" * 85)

    sort_col = "f1" if "f1" in metrics.columns else "accuracy"
    for _, row in metrics.sort_values(sort_col, ascending=False).iterrows():
        tag = "⭐" if row.get("is_best", 0) else ""
        print(
            f"  {row['model_name']:<23} "
            f"{row.get('accuracy', 0):>9.1f} "
            f"{row.get('precision', 0):>10.4f} "
            f"{row.get('recall', 0):>8.4f} "
            f"{row.get('f1', 0):>8.4f} "
            f"{row.get('roc_auc', 0):>8.4f} "
            f"{tag:>6}"
        )

    print("=" * 85)
    print("\nGhi chú:")
    print("  - Acc (%)   : % ngày dự đoán đúng xu hướng")
    print("  - Precision : Khi dự đoán Tăng, xác suất đúng")
    print("  - Recall    : % ngày thực sự Tăng được phát hiện")
    print("  - F1        : Harmonic mean của Precision & Recall")
    print("  - AUC       : Khả năng phân biệt Tăng/Giảm (ROC-AUC)")


# ================================================================
# 2. Backtesting (Mô phỏng chiến lược giao dịch)
# ================================================================
def backtest(initial_capital: float = 10_000_000.0) -> dict:
    """
    Mô phỏng chiến lược giao dịch dựa trên dự đoán xu hướng.

    Chiến lược:
      - Nếu model dự đoán TĂNG → Buy (giữ cổ phiếu)
      - Nếu model dự đoán GIẢM → Cash (không giữ cổ phiếu)
      - So sánh với Buy-and-Hold (giữ suốt)

    Giả định:
      - Phí giao dịch: 0.15% mỗi lần mua/bán (phù hợp thực tế VN)
      - Chỉ trade trên TEST SET (chronological split)
      - Không sử dụng dữ liệu tương lai

    Input tables: predictions, raw_prices
    Output: dict với các metrics hiệu suất
    """
    logger.info("=" * 60)
    logger.info("BACKTEST — Mô phỏng chiến lược giao dịch")
    logger.info("=" * 60)

    # Load predictions
    preds = read_table("predictions")
    prices = read_table("raw_prices")[["date", "close"]].sort_values("date").reset_index(drop=True)

    if preds.empty:
        logger.error("❌ Chưa có predictions. Chạy 'python -m models.predict' trước.")
        return {}

    if prices.empty:
        logger.error("❌ Chưa có raw_prices.")
        return {}

    # Merge predictions với giá thực tế
    preds["date"] = pd.to_datetime(preds["date"]).dt.strftime("%Y-%m-%d")
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")

    df = preds.merge(prices, on="date", how="inner").sort_values("date").reset_index(drop=True)

    if df.empty:
        logger.error("❌ Không khớp được predictions với giá.")
        return {}

    metrics = read_table("model_metrics")
    model_name = df["model_name"].iloc[0] if "model_name" in df.columns and not df.empty else None
    cutoff_date = None
    if not metrics.empty and model_name:
        model_metrics = metrics[metrics["model_name"] == model_name]
        if not model_metrics.empty and "val_end_date" in model_metrics.columns:
            val_end = model_metrics.iloc[0].get("val_end_date")
            if pd.notna(val_end):
                cutoff_date = pd.to_datetime(val_end)

    if cutoff_date is None:
        merged = read_table("merged_features")
        if not merged.empty and "date" in merged.columns:
            merged = merged.sort_values("date").reset_index(drop=True)
            val_end_raw = int(len(merged) * (TRAIN_RATIO + VAL_RATIO))
            if 0 < val_end_raw < len(merged):
                cutoff_date = pd.to_datetime(merged.iloc[val_end_raw - 1]["date"])

    if cutoff_date is not None:
        before_filter = len(df)
        df = df[pd.to_datetime(df["date"]) > cutoff_date].reset_index(drop=True)
        logger.info(
            f"  Chỉ backtest out-of-sample sau {cutoff_date.strftime('%Y-%m-%d')} "
            f"({len(df)}/{before_filter} predictions)"
        )

    # Xác định predicted trend từ predicted_proba
    if "predicted_proba" in df.columns:
        df["pred_trend"] = (df["predicted_proba"] >= 0.5).astype(int)
    elif "predicted_trend" in df.columns:
        df["pred_trend"] = df["predicted_trend"]
    else:
        logger.error("❌ Không tìm thấy cột predicted_proba hoặc predicted_trend.")
        return {}

    # Tính actual trend
    df["next_close"] = df["close"].shift(-1)
    df["actual_trend"] = (df["next_close"] > df["close"]).astype(int)
    df = df.dropna(subset=["next_close"]).reset_index(drop=True)

    if len(df) < 2:
        logger.error("❌ Không đủ dữ liệu out-of-sample để backtest.")
        return {}

    logger.info(f"  Backtesting trên {len(df)} phiên giao dịch")
    logger.info(f"  Từ {df['date'].iloc[0]} đến {df['date'].iloc[-1]}")

    # ── Tính equity curve ─────────────────────────────────────────
    TRANSACTION_FEE = 0.0015  # 0.15%

    capital_strategy = initial_capital
    capital_buyhold = initial_capital
    shares_strategy = 0.0
    shares_buyhold = initial_capital / df["close"].iloc[0]

    strategy_values = [initial_capital]
    buyhold_values = [initial_capital]
    in_position = False

    correct_preds = 0

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_close = df.iloc[i + 1]["close"]

        # Buy-and-Hold
        buyhold_values.append(shares_buyhold * next_close)

        # Strategy: mua khi predict Tăng, bán khi predict Giảm
        if row["pred_trend"] == 1 and not in_position:
            # Buy
            fee = capital_strategy * TRANSACTION_FEE
            shares_strategy = (capital_strategy - fee) / row["close"]
            capital_strategy = 0
            in_position = True
        elif row["pred_trend"] == 0 and in_position:
            # Sell
            capital_strategy = shares_strategy * row["close"]
            fee = capital_strategy * TRANSACTION_FEE
            capital_strategy -= fee
            shares_strategy = 0
            in_position = False

        if in_position:
            portfolio_val = shares_strategy * next_close
        else:
            portfolio_val = capital_strategy

        strategy_values.append(portfolio_val)

        # Đếm dự đoán đúng
        if row["pred_trend"] == row["actual_trend"]:
            correct_preds += 1

    # Giá trị cuối (nếu vẫn đang giữ)
    if in_position:
        final_val = shares_strategy * df.iloc[-1]["close"]
        strategy_values[-1] = final_val

    final_strategy = strategy_values[-1]
    final_buyhold = buyhold_values[-1]

    ret_strategy = (final_strategy / initial_capital - 1) * 100
    ret_buyhold = (final_buyhold / initial_capital - 1) * 100
    directional_acc = correct_preds / (len(df) - 1) * 100

    # Sharpe Ratio (simplified, assuming 252 trading days/year, risk-free=0)
    daily_returns = pd.Series(strategy_values).pct_change().dropna()
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)

    # Max Drawdown
    equity_curve = pd.Series(strategy_values)
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min() * 100

    logger.info(f"\n{'=' * 50}")
    logger.info("KẾT QUẢ BACKTEST")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Vốn ban đầu    : {initial_capital:>15,.0f} VND")
    logger.info(f"  Chiến lược     : {final_strategy:>15,.0f} VND  ({ret_strategy:+.2f}%)")
    logger.info(f"  Buy-and-Hold   : {final_buyhold:>15,.0f} VND  ({ret_buyhold:+.2f}%)")
    logger.info(f"  Direction Acc  : {directional_acc:.1f}%")
    logger.info(f"  Sharpe Ratio   : {sharpe:.3f}")
    logger.info(f"  Max Drawdown   : {max_drawdown:.1f}%")
    logger.info(f"{'=' * 50}")

    return {
        "initial_capital": initial_capital,
        "final_strategy": final_strategy,
        "final_buyhold": final_buyhold,
        "return_strategy_pct": ret_strategy,
        "return_buyhold_pct": ret_buyhold,
        "directional_accuracy": directional_acc,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown,
        "strategy_values": strategy_values,
        "buyhold_values": buyhold_values,
        "dates": df["date"].tolist(),
    }


def get_backtest_equity_curve() -> pd.DataFrame:
    """Chạy backtest và trả về DataFrame cho biểu đồ (dùng trong Streamlit)."""
    result = backtest()
    if not result:
        return pd.DataFrame()
    dates = result["dates"][:len(result["strategy_values"])]
    return pd.DataFrame({
        "date": dates,
        "strategy": result["strategy_values"][:len(dates)],
        "buy_and_hold": result["buyhold_values"][:len(dates)],
    })


if __name__ == "__main__":
    compare_models()
    print()
    backtest()
