"""
Dashboard dự đoán xu hướng TCB (Classification version).
═══════════════════════════════════════════════════════════
Chạy: streamlit run web/app.py

Hiển thị:
  - Xác suất tăng giá (gauge chart)
  - So sánh PhoBERT vs TF-IDF sentiment
  - Biểu đồ equity curve (backtest)
  - Bảng so sánh models (classification metrics)
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.connection import read_table, table_exists

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="TCB Trend Prediction",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Dự đoán Xu hướng Cổ phiếu TCB — Techcombank")
st.caption("Project Data Science | Classification: Dự đoán Tăng/Giảm")

# ================================
# Load Data
# ================================
@st.cache_data(ttl=60)
def load_all_data():
    data = {}
    if table_exists("raw_prices"):
        data["prices"] = read_table("raw_prices")
    if table_exists("predictions"):
        data["predictions"] = read_table("predictions")
    if table_exists("clean_news"):
        data["news"] = read_table("clean_news")
    if table_exists("model_metrics"):
        data["metrics"] = read_table("model_metrics")
    if table_exists("merged_features"):
        data["features"] = read_table("merged_features")
    return data


data = load_all_data()

if "prices" not in data or data["prices"].empty:
    st.warning("⚠️ Chưa có data giá. Chạy: `python -m data_collection.collect_prices`")
    st.stop()

prices = data["prices"].sort_values("date")

# ================================
# Header Metrics
# ================================
col1, col2, col3, col4, col5 = st.columns(5)

current_price = prices["close"].iloc[-1]
prev_price = prices["close"].iloc[-2] if len(prices) > 1 else current_price
change_pct = (current_price - prev_price) / prev_price * 100

col1.metric("Giá hiện tại", f"{current_price:,.0f} VND", f"{change_pct:+.2f}%")
col2.metric("Số phiên giao dịch", f"{len(prices):,}")

# Dự đoán xu hướng mới nhất
if "predictions" in data and not data["predictions"].empty:
    preds = data["predictions"].sort_values("date")
    latest = preds.iloc[-1]
    # Hỗ trợ cả schema cũ (predicted_price) và mới (predicted_proba/predicted_trend)
    if "predicted_proba" in preds.columns:
        proba = float(latest.get("predicted_proba", 0.5))
        trend = int(latest.get("predicted_trend", 0) if "predicted_trend" in preds.columns else (proba >= 0.5))
        trend_label = "📈 TĂNG" if trend == 1 else "📉 GIẢM"
        col3.metric(
            f"Dự báo ngày {latest.get('date', '?')}",
            trend_label,
            f"P(tăng) = {proba:.1%}",
            delta_color="normal" if trend == 1 else "inverse",
        )
    elif "predicted_price" in preds.columns:
        pred_price = latest.get("predicted_price", 0)
        col3.metric(f"Dự báo giá ({latest.get('date', '?')})", f"{pred_price:,.0f} VND")

# Best model info
if "metrics" in data and not data["metrics"].empty:
    sort_col = "f1" if "f1" in data["metrics"].columns else "accuracy"
    if sort_col not in data["metrics"].columns:
        sort_col = data["metrics"].columns[1]  # fallback
    best = data["metrics"].sort_values(sort_col, ascending=False).iloc[0]
    f1_val = best.get("f1", best.get("directional_accuracy", 0)) or 0
    acc_val = best.get("accuracy", 0) or 0
    col4.metric(
        f"Best Model ({best['model_name']})",
        f"F1: {f1_val:.4f}",
        f"Acc: {acc_val:.1f}%",
    )

    # Accuracy from known predictions
    if "predictions" in data and not data["predictions"].empty:
        p = data["predictions"]
        if "actual_trend" in p.columns and "predicted_trend" in p.columns:
            p_known = p.dropna(subset=["actual_trend"])
            if not p_known.empty:
                acc = (p_known["predicted_trend"] == p_known["actual_trend"]).mean() * 100
                col5.metric("Direction Accuracy", f"{acc:.1f}%", "trên test set")

st.divider()

# ================================
# Probability Gauge + Price Chart
# ================================
left, right = st.columns([1, 3])

with left:
    st.subheader("🎯 Xác suất Tăng giá")
    if "predictions" in data and not data["predictions"].empty:
        latest_proba = float(data["predictions"].sort_values("date").iloc[-1].get("predicted_proba", 0.5))
        # Gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_proba * 100,
            title={"text": "P(Tăng) %", "font": {"size": 14}},
            delta={"reference": 50, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#26a69a" if latest_proba >= 0.5 else "#ef5350"},
                "steps": [
                    {"range": [0, 40], "color": "#ffcdd2"},
                    {"range": [40, 60], "color": "#fff9c4"},
                    {"range": [60, 100], "color": "#c8e6c9"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10),
                            template="plotly_dark")
        st.plotly_chart(gauge, use_container_width=True)

        trend_color = "green" if latest_proba >= 0.5 else "red"
        trend_text = "📈 Dự đoán TĂNG" if latest_proba >= 0.5 else "📉 Dự đoán GIẢM"
        st.markdown(f"<h3 style='text-align:center; color:{trend_color}'>{trend_text}</h3>",
                    unsafe_allow_html=True)
    else:
        st.info("Chạy `python -m models.predict` để có dự đoán.")

with right:
    st.subheader("Biểu đồ giá TCB")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.75, 0.25],
        subplot_titles=("Giá TCB (candlestick)", "Khối lượng"),
    )
    fig.add_trace(go.Candlestick(
        x=prices["date"], open=prices["open"],
        high=prices["high"], low=prices["low"], close=prices["close"],
        name="Giá thực tế",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Overlay predicted trend arrows
    if "predictions" in data and not data["predictions"].empty:
        preds_df = data["predictions"]
        if "predicted_trend" in preds_df.columns:
            p_merged = preds_df.merge(prices[["date", "close"]], on="date", how="left")
            up_mask = p_merged["predicted_trend"] == 1
            down_mask = p_merged["predicted_trend"] == 0
            if up_mask.any():
                fig.add_trace(go.Scatter(
                    x=p_merged[up_mask]["date"],
                    y=p_merged[up_mask]["close"] * 0.995,
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="#26a69a", size=8),
                    name="Dự đoán Tăng",
                ), row=1, col=1)
            if down_mask.any():
                fig.add_trace(go.Scatter(
                    x=p_merged[down_mask]["date"],
                    y=p_merged[down_mask]["close"] * 1.005,
                    mode="markers",
                    marker=dict(symbol="triangle-down", color="#ef5350", size=8),
                    name="Dự đoán Giảm",
                ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=prices["date"], y=prices["volume"],
        name="Khối lượng", marker_color="rgba(100,150,200,0.4)",
    ), row=2, col=1)

    fig.update_layout(
        height=450, xaxis_rangeslider_visible=False,
        template="plotly_dark", showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ================================
# Model Comparison Table
# ================================
if "metrics" in data and not data["metrics"].empty:
    st.subheader("📊 So sánh Models (Classification Metrics)")
    metric_cols_map = {
        "model_name": "Model",
        "accuracy": "Acc (%)",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-Score",
        "roc_auc": "ROC-AUC",
        "is_best": "Best",
    }
    avail = [c for c in metric_cols_map if c in data["metrics"].columns]
    display_df = data["metrics"][avail].copy()
    display_df.columns = [metric_cols_map[c] for c in avail]
    if "Best" in display_df.columns:
        display_df["Best"] = display_df["Best"].map({1: "⭐", 0: ""})

    # Format floats
    for col in ["Acc (%)", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if col == "Acc (%)" else f"{x:.4f}" if pd.notna(x) else "—"
            )

    sort_col_ui = "F1-Score" if "F1-Score" in display_df.columns else "Acc (%)"
    display_df = display_df.sort_values(sort_col_ui, ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("Tiêu chí chọn best model: **F1-Score cao nhất** (phù hợp dữ liệu mất cân bằng)")

st.divider()

# ================================
# Backtesting Chart
# ================================
st.subheader("📈 Backtest — Mô phỏng chiến lược giao dịch")

if "predictions" in data and not data["predictions"].empty:
    try:
        from models.evaluate import get_backtest_equity_curve
        equity_df = get_backtest_equity_curve()
        if not equity_df.empty:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=equity_df["date"], y=equity_df["strategy"],
                name="🤖 Chiến lược AI", mode="lines",
                line=dict(color="#26a69a", width=2),
            ))
            fig_bt.add_trace(go.Scatter(
                x=equity_df["date"], y=equity_df["buy_and_hold"],
                name="📦 Buy-and-Hold", mode="lines",
                line=dict(color="#78909c", width=2, dash="dot"),
            ))
            fig_bt.update_layout(
                height=350, template="plotly_dark",
                title="Equity Curve (VND)",
                xaxis_title="Ngày", yaxis_title="Giá trị danh mục (VND)",
                legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_bt, use_container_width=True)
            st.caption(
                "Chiến lược: Mua khi P(tăng) ≥ 50%, chuyển Cash khi P(tăng) < 50%. "
                "Phí giao dịch: 0.15%/lần."
            )
        else:
            st.info("Không có đủ dữ liệu để backtest.")
    except Exception as e:
        st.info(f"Chạy `python -m models.predict` rồi thử lại. Chi tiết: {e}")
else:
    st.info("Chạy `python -m models.predict` để tạo predictions trước.")

st.divider()

# ================================
# Sentiment Comparison: PhoBERT vs TF-IDF
# ================================
st.subheader("📰 So sánh Sentiment — PhoBERT (DL) vs TF-IDF (Classical NLP)")

if "news" in data and not data["news"].empty:
    news_df = data["news"].copy()
    news_df["date"] = pd.to_datetime(news_df["date"])
    daily = news_df.groupby("date").agg(
        phobert_avg=("sentiment_score", "mean"),
        tfidf_avg=("tfidf_sentiment", "mean") if "tfidf_sentiment" in news_df.columns else ("sentiment_score", "mean"),
    ).reset_index()

    fig_sent = go.Figure()
    fig_sent.add_trace(go.Scatter(
        x=daily["date"], y=daily["phobert_avg"],
        name="PhoBERT (DL)", mode="lines",
        line=dict(color="#7c4dff", width=2),
    ))
    if "tfidf_avg" in daily.columns and "tfidf_sentiment" in news_df.columns:
        fig_sent.add_trace(go.Scatter(
            x=daily["date"], y=daily["tfidf_avg"],
            name="TF-IDF Lexicon (Classical)", mode="lines",
            line=dict(color="#ff6d00", width=2, dash="dash"),
        ))
    fig_sent.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_sent.update_layout(
        height=300, template="plotly_dark",
        title="Daily Sentiment Score (PhoBERT vs TF-IDF)",
        xaxis_title="Ngày", yaxis_title="Sentiment Score",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    # Recent news table
    st.subheader("📋 Tin tức gần đây")
    recent = news_df.sort_values("date", ascending=False).head(10)
    for _, row in recent.iterrows():
        score = row.get("sentiment_score", 0)
        tfidf = row.get("tfidf_sentiment", None)
        emoji = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
        tfidf_str = f" | TF-IDF: {tfidf:+.3f}" if tfidf is not None else ""
        st.markdown(f"{emoji} **{str(row['date'])[:10]}** — {row.get('title', '')} "
                    f"*(PhoBERT: {score:+.3f}{tfidf_str})*")
else:
    st.info("Chạy `python -m preprocessing.process_news` để phân tích sentiment.")

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.header("ℹ️ Thông tin")
    st.markdown("""
    **Project Data Science**
    - Dự đoán xu hướng giá TCB (Tăng/Giảm)
    - 3 nguồn data: giá, BCTC, tin tức
    - Baseline: Logistic Regression, Random Forest
    - Deep Learning: LSTM, GRU, Transformer
    - NLP: PhoBERT (DL) + TF-IDF Lexicon (Classical)
    """)

    st.divider()

    st.subheader("📦 Database Status")
    for table_name in [
        "raw_prices", "raw_finance", "raw_news",
        "clean_prices", "features_finance", "clean_news",
        "daily_news_embeddings", "merged_features",
        "predictions", "model_metrics",
    ]:
        if table_exists(table_name):
            try:
                count = len(read_table(table_name))
                st.markdown(f"✅ `{table_name}` — {count:,} rows")
            except Exception:
                st.markdown(f"⬜ `{table_name}` — empty")
        else:
            st.markdown(f"⬜ `{table_name}` — chưa có")

    st.divider()

    st.subheader("📖 Hướng dẫn chạy")
    st.code("""
# 1. Thu thập dữ liệu
python -m data_collection.collect_prices
python -m data_collection.collect_news

# 2. Tiền xử lý
python -m preprocessing.process_prices
python -m preprocessing.process_finance
python -m preprocessing.process_news
python -m preprocessing.merge_features

# 3. Train models
python -m models.train

# 4. Dự đoán
python -m models.predict

# 5. Backtest
python -m models.evaluate
""", language="bash")

    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
