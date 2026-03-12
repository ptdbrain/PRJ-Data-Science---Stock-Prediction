"""
Dashboard dự đoán giá TCB.
═══════════════════════════
Phụ trách: Thành viên F
Branch: feature/web-dashboard
Chạy: streamlit run web/app.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    page_title="TCB Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dự đoán giá cổ phiếu TCB — Techcombank")
st.caption("Project Data Science | Phase 1: Data tĩnh")

# ================================
# Load Data
# ================================
@st.cache_data(ttl=60)
def load_all_data():
    """Load tất cả data từ database."""
    data = {}

    if table_exists("raw_prices"):
        data['prices'] = read_table("raw_prices")
    if table_exists("predictions"):
        data['predictions'] = read_table("predictions")
    if table_exists("clean_news"):
        data['news'] = read_table("clean_news")
    if table_exists("model_metrics"):
        data['metrics'] = read_table("model_metrics")
    if table_exists("merged_features"):
        data['features'] = read_table("merged_features")

    return data


data = load_all_data()

# ================================
# Kiểm tra data
# ================================
if 'prices' not in data or data['prices'].empty:
    st.warning("⚠️ Chưa có data giá. Chạy: `python -m data_collection.collect_prices`")
    st.stop()

prices = data['prices'].sort_values('date')

# ================================
# Header Metrics
# ================================
col1, col2, col3, col4 = st.columns(4)

current_price = prices['close'].iloc[-1]
prev_price = prices['close'].iloc[-2] if len(prices) > 1 else current_price
change = (current_price - prev_price) / prev_price * 100

col1.metric("Giá hiện tại", f"{current_price:,.0f} VND", f"{change:+.2f}%")
col2.metric("Số phiên giao dịch", f"{len(prices):,}")

if 'predictions' in data and not data['predictions'].empty:
    preds = data['predictions'].sort_values('date')
    latest = preds.iloc[-1]
    pred_change = (latest['predicted_price'] - current_price) / current_price * 100
    col3.metric("Dự đoán mới nhất", f"{latest['predicted_price']:,.0f} VND", f"{pred_change:+.2f}%")

if 'metrics' in data and not data['metrics'].empty:
    best = data['metrics'].sort_values('mape').iloc[0]
    col4.metric(f"Best Model ({best['model_name']})", f"MAPE: {best['mape']:.2f}%")

st.divider()

# ================================
# Price Chart
# ================================
st.subheader("Biểu đồ giá và dự đoán")

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.05, row_heights=[0.75, 0.25],
    subplot_titles=("Giá TCB", "Khối lượng")
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=prices['date'], open=prices['open'],
    high=prices['high'], low=prices['low'], close=prices['close'],
    name='Giá thực tế',
    increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
), row=1, col=1)

# Predictions
if 'predictions' in data and not data['predictions'].empty:
    preds = data['predictions'].sort_values('date')
    fig.add_trace(go.Scatter(
        x=preds['date'], y=preds['predicted_price'],
        name='Dự đoán', mode='lines',
        line=dict(color='#ff9800', width=2, dash='dot')
    ), row=1, col=1)

# Volume
fig.add_trace(go.Bar(
    x=prices['date'], y=prices['volume'],
    name='Khối lượng', marker_color='rgba(100,150,200,0.4)'
), row=2, col=1)

fig.update_layout(
    height=600, xaxis_rangeslider_visible=False,
    template='plotly_dark', showlegend=True,
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
)
st.plotly_chart(fig, use_container_width=True)

# ================================
# Model Comparison
# ================================
if 'metrics' in data and not data['metrics'].empty:
    st.subheader("📊 So sánh Models")
    metrics_display = data['metrics'][
        ['model_name', 'rmse', 'mae', 'mape', 'directional_accuracy', 'is_best']
    ].copy()
    metrics_display.columns = ['Model', 'RMSE (VND)', 'MAE (VND)', 'MAPE (%)',
                                'Direction Acc (%)', 'Best']
    metrics_display['Best'] = metrics_display['Best'].map({1: '⭐', 0: ''})
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

# ================================
# Two columns: Predictions + News
# ================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔮 Lịch sử dự đoán")
    if 'predictions' in data and not data['predictions'].empty:
        display_df = data['predictions'].sort_values('date', ascending=False).head(20).copy()
        display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"{x:,.0f}")
        display_df['actual_price'] = display_df['actual_price'].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        st.dataframe(
            display_df[['date', 'model_name', 'predicted_price', 'actual_price']].rename(
                columns={'date': 'Ngày', 'model_name': 'Model',
                         'predicted_price': 'Dự đoán', 'actual_price': 'Thực tế'}
            ), use_container_width=True, hide_index=True
        )
    else:
        st.info("Chạy `python -m models.predict` để tạo predictions.")

with col_right:
    st.subheader("📰 Sentiment tin tức")
    if 'news' in data and not data['news'].empty:
        news = data['news'].sort_values('date', ascending=False).head(15)
        for _, row in news.iterrows():
            score = row.get('sentiment_score', 0)
            emoji = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
            st.markdown(f"{emoji} **{row['date']}** — {row['title']}")
    else:
        st.info("Chạy `python -m preprocessing.process_news` để phân tích sentiment.")

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.header("ℹ️ Thông tin")
    st.markdown("""
    **Project Data Science**
    - Dự đoán giá cổ phiếu TCB
    - 3 nguồn data: giá, BCTC, tin tức
    - Models: LSTM, GRU, Transformer
    """)

    st.divider()

    # Database status
    st.subheader("📦 Database Status")
    for table_name in ['raw_prices', 'raw_finance', 'raw_news',
                       'clean_prices', 'clean_finance', 'clean_news',
                       'merged_features', 'predictions', 'model_metrics']:
        if table_exists(table_name):
            try:
                count = len(read_table(table_name))
                st.markdown(f"✅ `{table_name}` — {count:,} rows")
            except Exception:
                st.markdown(f"⬜ `{table_name}` — empty")
        else:
            st.markdown(f"⬜ `{table_name}` — chưa có")

    st.divider()
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
