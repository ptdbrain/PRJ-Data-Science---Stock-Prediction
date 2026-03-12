# 📈 TCB Stock Price Prediction

Hệ thống dự đoán giá cổ phiếu **TCB (Techcombank)** sử dụng Deep Learning, kết hợp 3 nguồn dữ liệu: giá lịch sử, báo cáo tài chính, và tin tức (sentiment analysis).

## 🏗️ Kiến trúc

```
Data Collection → Preprocessing → Feature Engineering → Model Training → Web Dashboard
(3 nguồn)        (clean + enrich)  (merge 3 bảng)       (LSTM/GRU/Trans.)  (Streamlit)
```

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/HuuKhanh19/Project-Data-Science.git
cd Project-Data-Science

# 2. Setup môi trường
bash setup_env.sh
# HOẶC thủ công:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Tạo database
python -m database.schema

# 4. Thu thập data (mỗi người chạy phần mình)
python -m data_collection.collect_prices
python -m data_collection.collect_finance
python -m data_collection.collect_news

# 5. Tiền xử lý
python -m preprocessing.process_prices
python -m preprocessing.process_finance
python -m preprocessing.process_news
python -m preprocessing.merge_features

# 6. Train models
python -m models.train

# 7. Predict
python -m models.predict

# 8. Xem dashboard
streamlit run web/app.py
```

## 👥 Phân công

| Thành viên | Nhiệm vụ | Branch |
|-----------|-----------|--------|
| A | Giá cổ phiếu: thu thập + tiền xử lý + indicators | `feature/collect-prices`, `feature/process-prices` |
| B | BCTC: thu thập + tiền xử lý + ratios | `feature/collect-finance`, `feature/process-finance` |
| C | Tin tức: thu thập + tiền xử lý + sentiment | `feature/collect-news`, `feature/process-news` |
| D | Merge features + Model LSTM | `feature/merge-features`, `feature/model-lstm` |
| E | Model GRU + Transformer | `feature/model-gru`, `feature/model-transformer` |
| F | Database + Web Dashboard + DevOps | `feature/web-dashboard` |

## 📁 Cấu trúc

```
├── config/settings.py          # Cấu hình tập trung
├── database/
│   ├── connection.py           # Kết nối DB dùng chung
│   └── schema.py               # Tạo tables
├── data_collection/            # Thu thập data thô
├── preprocessing/              # Tiền xử lý + merge
├── models/
│   ├── base_model.py           # Base class chung
│   ├── lstm_model.py           # LSTM
│   ├── gru_model.py            # GRU
│   ├── transformer_model.py    # Transformer
│   ├── train.py                # Train + so sánh
│   └── predict.py              # Predict
└── web/app.py                  # Streamlit dashboard
```

## 📊 Database Tables

| Table | Mô tả | Phụ trách |
|-------|--------|-----------|
| `raw_prices` | Giá OHLCV gốc | A |
| `raw_finance` | BCTC gốc | B |
| `raw_news` | Tin tức gốc | C |
| `clean_prices` | Giá + technical indicators | A |
| `clean_finance` | BCTC + financial ratios | B |
| `clean_news` | Tin tức + sentiment | C |
| `merged_features` | Tổng hợp → input model | D |
| `predictions` | Kết quả dự đoán | D, E |
| `model_metrics` | So sánh models | D, E |

## 🔀 Git Workflow

1. Checkout `develop`, pull mới nhất
2. Tạo branch `feature/tên-task`
3. Code + commit thường xuyên
4. Push → tạo Pull Request → develop
5. 1 người review + approve → merge
